import numpy as np
import yaml, caffe
from other import clip_boxes
from anchor import AnchorText
from rpn.generate_anchors import locate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv

class ProposalLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        # self._feat_stride = layer_params['feat_stride']
        # self.anchor_generator=AnchorText()
        # self._num_anchors = self.anchor_generator.anchor_num

        height, width = bottom[0].data.shape[-2:]
        self._feat_stride = layer_params['feat_stride']

        anchor_scales = layer_params.get('scales', (8, 16, 32))
        #self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._anchors = locate_anchors((height, width), self._feat_stride)

        self._num_anchors = self._anchors.shape[0]
        top[0].reshape(1, 4)
        top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0]==1, \
            'Only single item batches are supported'

        scores = bottom[0].data[:, self._num_anchors:, :, :]

        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)






        # anchors=self.anchor_generator.locate_anchors((height, width), self._feat_stride)
        #
        # scores=scores.transpose((0, 2, 3, 1)).reshape(-1, 1)
        # bbox_deltas=bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 2))
        #
        # proposals=self.anchor_generator.apply_deltas_to_anchors(bbox_deltas, anchors)

        # clip the proposals in excess of the boundaries of the image
        proposals=clip_boxes(proposals, im_info[:2])

        blob=proposals.astype(np.float32, copy=False)
        top[0].reshape(*(blob.shape))
        top[0].data[...]=blob

        top[1].reshape(*(scores.shape))
        top[1].data[...]=scores

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
