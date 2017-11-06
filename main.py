from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from proto_weight_decomp import proto_decomp,weight_decomp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add caffe to PYTHONPATH
add_path('/home/bryant/msrpn/python/')

proto_input1 = "/home/bryant/compress/trainval_1st.prototxt"
proto_input2 = "/home/bryant/compress/trainval_2nd.prototxt"
proto_input3 = "/home/bryant/compress/deploy.prototxt"
weight_input = "/home/bryant/compress/caltech_train_2nd_iter_25000.caffemodel"

proto_output1 = "/home/bryant/msrpn/compress/compressed-conv5-trainval_1st.prototxt"
proto_output2 = "/home/bryant/msrpn/compress/compressed-conv5-trainval_2nd.prototxt"
proto_output3 = "/home/bryant/msrpn/compress/compressed_deploy.prototxt"
weight_output = "/home/bryant/msrpn/compress/compressed-conv5-pretrained.caffemodel"



compressed_layer = {'conv1_2': (32,32), 'conv2_1': (32,32), 'conv2_2': (32, 32),
           'conv3_1': (64,64), 'conv3_2': (64,64), 'conv3_3': (64, 64),
           'conv4_1': (64,128), 'conv4_2': (128,128), 'conv4_3': (128, 128),
           'conv5_1': (128,128), 'conv5_2': (128,128), 'conv5_3': (128, 128),'roi_c1': (128, 128),'loss1_conv': (128, 128), 'fc6': 300}

if __name__ == "__main__":
    proto_decomp(proto_input1, weight_input, proto_output1, compressed_layer)
    proto_decomp(proto_input2, weight_input, proto_output2, compressed_layer)
    proto_decomp(proto_input3, weight_input, proto_output3, compressed_layer)
    weight_decomp(proto_input2, weight_input, proto_output2, weight_output, compressed_layer)	
