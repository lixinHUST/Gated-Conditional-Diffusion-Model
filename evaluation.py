import torch
import numpy as np
from PIL import Image
import os
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import math
import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import lpips
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import SimpleITK as sitk

def compute_fid(data_folder1, data_folder2,device='cuda:0'):
    from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)
    ref_features = get_folder_features(data_folder1, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=32, device=torch.device(device),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
    a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
    
    gen_features = get_folder_features(data_folder2, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=32, device=torch.device(device),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
    ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
    print(f"fid={score_fid_a2b:.4f}")

def img2mask(img,T=15):
    img[img<T]=0
    img[img>0]=1
    mask=img.copy()
    kernel = np.ones((5,5),np.uint8)  
    mask = cv2.erode(mask,kernel,iterations = 1)
    
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Find the maximum area and fill it in
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mask, [contours[k]], 0)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    mask = cv2.erode(mask,kernel,iterations = 1)
    return mask

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

def Read_nifti(img_path):
    img_sitk = sitk.ReadImage(img_path)
    return sitk.GetArrayFromImage(img_sitk)

def seg_inference(data_path, mask_path, medsam_model, device):
    img_3c= np.array(Image.open(data_path).resize((256,256), Image.BILINEAR).convert('RGB'))
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    bbox_arr=Read_nifti(mask_path)
    H, W = bbox_arr.shape
    xmin = np.where(bbox_arr==1)[0].min()
    xmax = np.where(bbox_arr==1)[0].max()
    ymin = np.where(bbox_arr==1)[1].min()
    ymax = np.where(bbox_arr==1)[1].max()

    box_np = np.array([[ymin, xmin, ymax, xmax]])
    box_256 = box_np / np.array([W, H, W, H]) * 256
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, 256, 256)
    return medsam_seg,img_3c,box_256

"""
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc 
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) 
        return meanAcc 
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) 
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)  
        IoU = intersection / union 
        mIoU = np.nanmean(IoU) 
        return IoU,mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 
def compute_IoU_PA(mask_path,fake_path,box_path,device='cuda:0'):
    mask_dict={}
    for name in os.listdir(box_path):
        maskid=name.split('_bbox')[0]
        if maskid not in mask_dict.keys():
            mask_dict[maskid]=[name]
        else:
            mask_dict[maskid].append(name)
    
    MedSAM_CKPT_PATH = "pre_trained/medsam/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    savedir=fake_path+'_segs'
    os.makedirs(savedir,exist_ok=True)
    metric = SegmentationMetric(3) 
    for name in os.listdir(mask_path):
        mask = np.array(Image.open(os.path.join(mask_path,name)).resize((256,256),Image.NEAREST).convert('L'))
        if (mask==2).sum() > 0:
                save=os.path.join(savedir,name)
                img_path=os.path.join(fake_path,name)
                img= np.array(Image.open(img_path).resize((256,256),Image.BILINEAR).convert('L'))
                seg_mask = img2mask(img)
                maskid=name.split('.')[0]
                for box_name in mask_dict[maskid]:
                    boxpath=os.path.join(box_path,box_name)
                    medsam_seg,_,box=seg_inference(img_path,boxpath,medsam_model,device)
                    seg_mask[medsam_seg==1]=2
                metric.addBatch(mask, seg_mask)
                Image.fromarray(seg_mask.astype(np.uint8)).save(save)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    Iou,mIoU = metric.meanIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :') 
    print(cpa)
    print('mpa is : %f' % mpa)
    print('Iou is :') 
    print(Iou)
    print('mIoU is : %f' % mIoU)

if __name__ == '__main__':

    real_path='DATA_FOLDER/images/test'
    mask_path='DATA_FOLDER/masks/test'
    box_path='DATA_FOLDER/bbox/test'

    device = "cuda:0"
    fake='test_results'
    compute_fid(real_path,fake,device=device)
    compute_IoU_PA(mask_path,fake,box_path,device=device)
   
    