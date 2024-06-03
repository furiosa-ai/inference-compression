"""
pytoch/caffe2 backend via onnx
https://pytorch.org/docs/stable/onnx.html
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

#import caffe2.python.onnx.backend
import onnx
import torch  # needed to get version and cuda setup

import backend

from torchvision.models import resnet50


class BackendPytorch(backend.Backend):
    def __init__(self):
        super(BackendPytorch, self).__init__()
        self.sess = None
        self.model = None
        self.lock = Lock()

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = resnet50()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

        #self.model = onnx.load(model_path)

        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

        # prepare the backend
        device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        #self.sess = onnx.backend.prepare(self.model, device)
        return self

    def predict(self, feed):
        # self.lock.acquire()
        # res = self.sess.run(feed)
        # self.lock.release()
        
        return self.model(torch.Tensor(feed['image'])).detach().numpy()
    
        # device = torch.device("cuda")
        # self.model.to(device)
        # return self.model(torch.Tensor(feed['image']).to(device)).detach().numpy()
