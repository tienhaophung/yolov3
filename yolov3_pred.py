from sys import platform
import torch
import torchvision
from torchvision import transforms

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

def prepare_data(images, img_size=512):
    """
    Prepare data for prediction by resizing images, converting BGR to RGB, transpose channels to 3xHxW

    Args:
    ---
    - images (list): a list of images in BGR format
    - img_size (int): image size to scale 

    Returns:
    ---
    - img0_list (list): a list of original images in BGR format. Each image has shape (HxWx3)
    - img_list (list): a list of scaled images in RGB format. Each image has shape (3xHxW)
    - images_tensor (tensor): torch tensor with format: (n, 3, h, w)
    """
    images_tensor = torch.zeros((len(images), 3, img_size, img_size))
    images_list = []
    img0_list = []
    # img_list = []
    for i in range(len(images)):
        img0 = images[i]

        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]
        # print(img.shape)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        if img.shape[0] == 3:
            img = np.expand_dims(img, axis=0)

        # Store to list
        img0_list.append(img0)
        # img_list.append(img)
        images_list.append(img) 

    images_tensor = torch.from_numpy(np.concatenate(images_list, 0))

    return img0_list, images_tensor

    
class YOLOv3_detection():
    def __init__(self,
                cfg="cfg/yolov3-spp-custom.cfg",  
                class_names="data/custom/classes.names",
                weights="weights/yolov3-best-100eps-1e-3-sgd.pt", 
                img_size=512, 
                conf_thres=0.3,
                iou_thres=0.15,
                view_img=True, 
                classes=None, # filter by class
                half=False, # half precision FP16 instead FP32 to speed up training
                augment=False, # using mosaic augmentaion for inference
                agnostic_nms=False, # class-agnostic NMS
                batch_size=8, 
                # save_txt=False, # Save results to .txt file
                # output="output", 
                device=""):
        self.cfg = cfg
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.view_img = view_img
        self.classes = classes
        # self.save_txt = save_txt
        # self.output = output
        self.augment = augment
        self.batch_size = batch_size
        self.agnostic_nms = agnostic_nms

        # Initialize device and make output dir
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else device)
        # if os.path.exists(self.output):
        #     shutil.rmtree(self.output)  # delete output folder
        # os.makedirs(self.output)  # make new output folder

        # Initialize model
        self.model = Darknet(self.cfg, self.img_size)
        
        # Load weights
        attempt_download(self.weights)
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        # # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        #     modelc.to(device).eval()

        # Eval mode
        self.model.to(self.device).eval()
        
        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Export mode
        if ONNX_EXPORT:
            self.model.fuse()
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            f = self.weights.replace(self.weights.split('.')[-1], 'onnx')  # *.onnx filename
            torch.onnx.export(self.model, img, f, verbose=False, opset_version=11)

            # Validate exported model
            import onnx
            self.model = onnx.load(f)  # Load the ONNX model
            onnx.checker.check_model(self.model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(self.model.graph))  # Print a human readable representation of the graph
            return
        

        # Half precision
        self.half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()
        
        # Get names and colors
        self.class_names = load_classes(class_names)


    def detect(self, images, 
                     # save_img=False
              ):
        """
        Detect bboxs in images

        Args:
        ---
        - images (list): a list of images in BGR format
        
        Returns:
        ---
        - detections (np.ndarray): results of human detection for images with format:
            (#images, #bboxs, 6). Where 6 values are (x1, y1, x2, y2, obj_conf, class)
        - img0_list (list): a list of original images that are painted with detection results.
        """
        # # Set Dataloader
        # vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = True
        #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=img_size)
        # else:
        #     save_img = True
        #     dataset = LoadImages(source, img_size=img_size)

        # Get colors
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        pred_list = []

        # Run inference
        t0 = time.time()
        _ = self.model(torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)) if device.type != 'cpu' else None  # run once
        
        # Prepare data
        img0_list, images_tensor = prepare_data(images, img_size=self.img_size)
        detections = []
        for t in range(0, len(img0_list), self.batch_size):
            input_images = images_tensor[t:t+self.batch_size].to(self.device)
            input_images = input_images.half() if self.half else input_images.float()  # uint8 to fp16/32
            # print(input_images.shape)
            # scale image
            input_images /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if input_images.ndimension() == 3:
            #     input_images = input_images.unsqueeze(0)
            # print(input_images.shape)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = self.model(input_images, augment=self.augment)[0] # (#images, #objects, 6)
            t2 = torch_utils.time_synchronized()
            # print(pred.shape)

            # to float
            if self.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                    multi_label=False, classes=self.classes, agnostic=self.agnostic_nms)
            detections.append(pred)

            # # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

        detections = np.concatenate(detections, 0)
        print(detections.shape)
        # print(images_tensor.shape)
        # for img_id, pred in enumerate(detections): # iterate through images' results
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            img = images_tensor[i]
            img0 = img0_list[i]
            if pred == None:
                continue
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i]
            # else:
            #     p, s, im0 = path, '', im0s

            # save_path = str(Path(out) / Path(p).name)
            s = '%gx%g ' % img.shape[1:]  # print string
            # print(s)
            if det is not None and len(det):
                print(det.shape)
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape[:-1]).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.class_names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    # if save_txt:  # Write to file
                    #     with open(save_path + '.txt', 'a') as file:
                    #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if self.view_img: # or save_img # Add bbox to image
                        label = '%s %.2f' % (self.class_names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
                        img0_list[i] = img0 # Update drawing

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if self.view_img:
                cv2.imshow("Image", img0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer

            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            #         vid_writer.write(im0)

        # if self.save_txt or save_img:
        #     print('Results saved to %s' % os.getcwd() + os.sep + out)
        #     if platform == 'darwin':  # MacOS
        #         os.system('open ' + out + ' ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))

        return detections, img0_list

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(device)

    img_path = r"C:\Users\User\Desktop\simple-HRNet\hand_images\hand4.jpg"
    img = cv2.imread(img_path)

    # Init model
    yolov3_detector = YOLOv3_detection()
    # Detect
    detections, img_list = yolov3_detector.detect([img])

    # cv2.imshow("Frame", img_list[0])
    cv2.waitKey(0)

