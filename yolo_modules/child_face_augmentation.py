train_tf = A.Compose([
    A.SmallestMaxSize(max_size=720),
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.Affine(translate_percent=(-0.1,0.1), rotate=(-10,10),
             shear=(-5,5), scale=(0.9,1.1), p=0.5),
    A.ColorJitter(0.3,0.3,0.3,0.05, p=0.4),
    A.OneOf([
        A.GaussNoise(10,30),
        A.ISONoise(0.01,0.05),
        A.ImageCompression(70,95)
    ], p=0.4),
    A.OneOf([
        A.MotionBlur(5),
        A.GaussianBlur((3,7))
    ], p=0.2),
    A.CoarseDropout(max_holes=2, max_height=0.1, max_width=0.1, p=0.2),
    A.Resize(640,640),
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1,
                             label_fields=['class_labels']))
