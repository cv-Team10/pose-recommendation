import os

class Config:
    def __init__(self):
        
        self.device = 'cpu' # ['cuda:0', 'cpu']
        self.model_type = 'vae' # ['pose_resnet', 'resnet', 'transformer', 'vae']

        if self.model_type == 'pose_resnet':
            self.image_resize = (256, 192)
        if self.model_type == 'resnet' or self.model_type == 'transformer' or self.model_type == 'vae':
            self.image_resize = (224, 224)

        self.num_epochs = 1000
        self.learning_rate = 2e-4
        self.batch_size = 16
        self.weight_decay = 5e-4
        self.num_workers = 4

        self.checkpoint_dir = 'checkpoints/experiment1'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)