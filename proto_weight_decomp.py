import caffe
from caffe.proto import caffe_pb2
from google import protobuf
import numpy as np

import tucker

def proto_decomp(proto_input, weight_input, proto_output, lra_map):
    net = caffe_pb2.NetParameter()
    fin = open(proto_input, 'r')
    protobuf.text_format.Merge(fin.read(), net)
    fin.close()
    caffe_net = caffe.Net(proto_input, weight_input, caffe.TEST)
    
    layer_list = list(net.layer)

    for layer in net.layer:
        if layer.name in lra_map:
            if layer.type == 'Convolution':
                layer_in = caffe_net.params[layer.name][0].data.shape[1]
                layer_out = caffe_net.params[layer.name][0].data.shape[0]

                conv_layers = [caffe_pb2.LayerParameter() for _ in range(3)]
                num_outputs = list(lra_map[layer.name]) + [layer_out]
                
                for i in range(3):
                    conv_layers[i].CopyFrom(layer)

                    if i != 1:
                        conv_layers[i].convolution_param.kernel_size[0] = 1
                        conv_layers[i].convolution_param.pad[0] = 0

                    conv_layers[i].convolution_param.num_output = num_outputs[i]
                    conv_layers[i].name = layer.name + '_lra_' + 'abc'[i]
                    conv_layers[i].top.__delslice__(0, len(conv_layers[i].top))
                    conv_layers[i].bottom.__delslice__(0, len(conv_layers[i].bottom))
                    conv_layers[i].top.append(layer.name + '_lra_' + 'abc'[i])
                    conv_layers[i].bottom.append(layer.name + '_lra_' + '-ab'[i])


                if layer_in == lra_map[layer.name][0]:
                    conv_layers = conv_layers[1:]
                if layer_out == lra_map[layer.name][1]:
                    conv_layers = conv_layers[:-1]

                conv_layers[0].bottom.__delslice__(0,1)
                conv_layers[0].bottom.MergeFrom(layer.bottom)
                conv_layers[-1].top.__delslice__(0,1)
                conv_layers[-1].top.MergeFrom(layer.top)

                if len(conv_layers) != 1:
                    idx = layer_list.index(layer)
                    layer_list.remove(layer)
                    for new_layer in conv_layers[::-1]:
                        layer_list.insert(idx, new_layer)
                else:
                    print ('%s has exactly same dimension as original layer. Abort lra for this layer' % layer.name)
                    
            elif layer.type == 'InnerProduct':
                layer_in = caffe_net.params[layer.name][0].data.shape[1]
                layer_out = caffe_net.params[layer.name][0].data.shape[0]
                if lra_map[layer.name] == layer_out:
                    print ('%s has exactly same dimension as original layer. Abort svd for this layer' % layer.name)
                    continue
                ip_layers = [caffe_pb2.LayerParameter() for _ in range(2)]
                num_outputs = [lra_map[layer.name], layer.convolution_param.num_output]
                for i in range(2):
                    ip_layers[i].CopyFrom(layer)
                    ip_layers[i].name = layer.name + '_svd_' + 'ab'[i]
                    
                ip_layers[0].inner_product_param.num_output = lra_map[layer.name]
                ip_layers[0].top.__delslice__(0, len(ip_layers[0].top))
                ip_layers[1].bottom.__delslice__(0, len(ip_layers[1].bottom))

                ip_layers[0].top.append(layer.name + '_svd_a')
                ip_layers[1].bottom.append(layer.name + '_svd_a')


                idx = layer_list.index(layer)
                layer_list.remove(layer)
                for new_layer in ip_layers[::-1]:
                    layer_list.insert(idx, new_layer)
            else:
                print ('Error processing layer $s: Type %s is not supported.' % (layer.name, layer.type))
    old_num = len(net.layer)
    net.layer.extend(layer_list)
    net.layer.__delslice__(0, old_num)
    fout =  open(proto_output, 'w')
    fout.write(str(net))
    fout.close()

def weight_decomp(proto_input, weight_input, proto_output, weight_output, lra_map):
    input_net = caffe_pb2.NetParameter()
    output_net = caffe_pb2.NetParameter()
    fin = open(weight_input, 'rb')
    input_net.ParseFromString(fin.read())
    fin.close()
    fin = open(proto_output, 'r')
    protobuf.text_format.Merge(fin.read(), output_net)
    fin.close()
    
    caffe_net = caffe.Net(proto_output, caffe.TEST)
    

    for layer in input_net.layer:
        if layer.name in lra_map:
            if layer.type == 'Convolution':
                if layer.name in caffe_net.params:
                    print ('%s has exactly same dimension as original layer. Abort svd for this layer' % layer.name)
                    for i, param in enumerate(caffe_net.params[layer.name]):
                        param.data[:] = np.array(layer.blobs[i].data).reshape(param.data.shape)
                    continue
                tensor = np.array(layer.blobs[0].data).reshape(layer.blobs[0].shape.dim)
                w = [None for _ in range(3)]
                w[1], w[2], w[0] = tucker.HOOI(tensor, *lra_map[layer.name])
                layer_names = [layer.name + '_lra_' + s for s in 'abc']

                for i in range(3):
                    if layer_names[i] in caffe_net.params:
                        print (layer_names[i], caffe_net.params[layer_names[i]][0].data.shape)

                for i in range(3):
                    if layer_names[i] in caffe_net.params:
                        if len(w[i].shape) == 2:
                            w[i] = w[i][:,:, np.newaxis, np.newaxis]
                        caffe_net.params[layer_names[i]][0].data[:] = w[i]
                for i in range(2,-1,-1):
                    if layer_names[i] in caffe_net.params:
                        caffe_net.params[layer_names[i]][1].data[:] = layer.blobs[1].data
                        break

            elif layer.type == 'InnerProduct':
                if layer.name in caffe_net.params:
                    print ('%s has exactly same dimension as original layer. Abort svd for this layer' % layer.name)
                    for i, param in enumerate(caffe_net.params[layer.name]):
                        print (layer.blobs[i].shape.dim,param.data.shape)
                        param.data[:] = np.array(layer.blobs[i].data).reshape(param.data.shape)
                    continue
                tensor = np.array(layer.blobs[0].data).reshape(layer.blobs[0].shape.dim)
                Us, sVh = tucker.trunc_svd(tensor, lra_map[layer.name])
                caffe_net.params[layer.name + '_svd_a'][0].data[:] = sVh
                caffe_net.params[layer.name + '_svd_b'][0].data[:] = Us
                caffe_net.params[layer.name + '_svd_b'][1].data[:] = layer.blobs[1].data
            else:
                print ('Error processing layer $s: Type %s is not supported.' % (layer.name, layer.type))
                  
        else:
            if layer.name in caffe_net.params:
                for i, param in enumerate(caffe_net.params[layer.name]):
                    param.data[:] = np.array(layer.blobs[i].data).reshape(param.data.shape)

    caffe_net.save(weight_output)
