import os
# import sys
"""
example:

data
└── avenue
workspace
├── anomaly_scores
│   └── avenue.pkl
├── logs
│   ├── avenue
│   │   ├── events.out.tfevents.1571121305.cc-B250M-Gaming5
│   └── log.txt
├── models
│   ├── CAE_avenue
│   │   ├── avenue.ckpt.data-00000-of-00001
│   │   ├── avenue.ckpt.index
│   │   ├── avenue.ckpt.meta
│   │   └── checkpoint
│   ├── ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
│   │   ├── checkpoint
│   │   ├── frozen_inference_graph.pb
│   │   ├── model.ckpt.data-00000-of-00001
│   │   ├── model.ckpt.index
│   │   ├── model.ckpt.meta
│   │   ├── pipeline.config
│   │   └── saved_model
│   │       ├── saved_model.pb
│   │       └── variables
│   └── svm
│       └── avenue.m
"""

class Paths(object):
    # 组织项目用到的文件和目录
    def __init__(self):
        self._SAMPLE_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir)
        self._WORKSPACE_DIR_PATH = os.path.join(self._SAMPLE_ROOT, 'workspace')
        self._DATASET_DIR_PATH = os.path.join(self._SAMPLE_ROOT, 'data')

    def set_dataset_dir_path(self, dataset_dir):
        self._DATASET_DIR_PATH = dataset_dir

    def get_dataset_dir_path(self):
        return self._DATASET_DIR_PATH

    def set_workspace_dir_path(self, workspace_dir):
        self._WORKSPACE_DIR_PATH = workspace_dir

    def get_workspace_dir_path(self):
        return self._WORKSPACE_DIR_PATH

    # Fixed dir
    def get_sample_root(self):
        return self._SAMPLE_ROOT

    def get_logs_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'logs')

    def get_model_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'models')

    def get_model_detection_dir_path(self):
        ssd_model_dir = 'ssd_resnet50_v1_fpn_shared_box_predictor_' + \
            '640x640_coco14_sync_2018_07_03'
        return os.path.join(self.get_model_dir_path(), ssd_model_dir)

    def get_model_frozen_graph_path(self):
        ssd_frozen_graph_file = 'frozen_inference_graph.pb'
        return os.path.join(self.get_model_detection_dir_path(),
                            ssd_frozen_graph_file)

    def get_model_detection_label_path(self):
        return os.path.join(self.get_sample_root(),
                            'object_detection/data/mscoco_lable_map.pbtxt')

    def get_model_svm_dir_path(self):
        return os.path.join(self.get_model_dir_path(), 'svm')

    def get_model_cae_dir_path(self):
        return os.path.join(self.get_model_dir_path(), 'CAE_')

    def get_anomaly_scores_pickle_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'anomaly_scores')


PATHS = Paths()

if __name__ == "__main__":
    path = Paths()
    # path.set_workspace_dir_path('./experiments/')
    print(path.get_workspace_dir_path())
    print(path.get_sample_root())
