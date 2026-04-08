import torch
import monai
from monai.inferers import sliding_window_inference
from torch import nn
from tqdm import tqdm
import numpy as np
from medpy import metric
from config import Config
from dataset import LAHeart, PancreasNIH, BraTS2019, BraTS2024
from networks.vnet import VNet
from networks.unet3D import UNet3D
import math
from skimage.measure import label

'''

    This file contains functions that perform the validation loop for
    BraTS 2019, BraTS-GLI 2024, Generic Binary 3D and ACDC 2D.  If you 
    are using a different dataset you have to write your own custom 
    validation function here.

    Each function has the following parameters:

        - model : the model being used for inferencing
        - best_performance : the performance of the last best performing checkpoint
        - validation_dataloader : the dataloader for the validation split
        - config : an instance of the Config class

    Each function returns 'best_performance' 

    You can also use these functions for evaluating the model on the test set.
    Just pass the testing dataloader instead of the validation dataloader.


'''

def compute_safe_dice(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0

    return metric.binary.dc(pred, gt)


def compute_safe_iou(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0

    return metric.binary.jc(pred, gt)


def compute_safe_hd95(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 0.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0

    return metric.binary.hd95(pred, gt)


def validation_loop_brats2019(model, best_performance, validation_dataloader, config : Config):

    model.eval()

    results = {
        "WT": {"Dice": [], "IoU": [], "HD95": []},
        "TC": {"Dice": [], "IoU": [], "HD95": []},
        "ET": {"Dice": [], "IoU": [], "HD95": []},
        "PTE": {"Dice": [], "IoU": [], "HD95": []},   
        "NET": {"Dice": [], "IoU": [], "HD95": []},
    }

    # This collects all results on ET, PTE and NET i.e. the base classes
    # from which the WT, TC and ET regions are computed. This is to get
    # the average across these base classes.
    results_base = {"Dice": [], "IoU": [], "HD95": []}

    with torch.no_grad():

        for image, mask in tqdm(validation_dataloader):

            image = image.cuda()
            mask = mask.cuda()

            def predictor(x):
                out = model(x)
                return out["final_output"]

            prediction = sliding_window_inference(
                image.float(),
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=predictor,
                overlap=0.5,
                mode="gaussian"
            )

            prediction = torch.argmax(prediction, dim=1)

            gt = mask.cpu().numpy()[0]
            pred = prediction.cpu().numpy()[0]

            net_gt = (gt == 1)
            pte_gt = (gt == 2)
            et_gt  = (gt == 3)

            net_pred = (pred == 1)
            pte_pred = (pred == 2)
            et_pred  = (pred == 3)

            tc_gt = np.logical_or(net_gt, et_gt)
            wt_gt = np.logical_or(tc_gt, pte_gt)

            tc_pred = np.logical_or(net_pred, et_pred)
            wt_pred = np.logical_or(tc_pred, pte_pred)

            region_map = {
                "WT": (wt_pred, wt_gt),
                "TC": (tc_pred, tc_gt),
                "ET": (et_pred, et_gt),
                "PTE": (pte_pred, pte_gt),
                "NET": (net_pred, net_gt),
            }

            case_sub_dice = []
            case_sub_iou = []
            case_sub_hd = []

            for region, (pred_r, gt_r) in region_map.items():

                dice = compute_safe_dice(pred_r, gt_r)
                iou  = compute_safe_iou(pred_r, gt_r)
                hd   = compute_safe_hd95(pred_r, gt_r)

                results[region]["Dice"].append(dice * 100)
                results[region]["IoU"].append(iou * 100)
                results[region]["HD95"].append(hd)

                if region in ["PTE", "NET", "ET"]:
                    case_sub_dice.append(dice)
                    case_sub_iou.append(iou)
                    case_sub_hd.append(hd)

            results_base["Dice"].append(np.mean(case_sub_dice) * 100)
            results_base["IoU"].append(np.mean(case_sub_iou) * 100)
            results_base["HD95"].append(np.mean(case_sub_hd))

    for region in results:
        for metric_name in results[region]:
            results[region][metric_name] = np.array(results[region][metric_name])

    for metric_name in results_base:
        results_base[metric_name] = np.array(results_base[metric_name])

    print("\nResults:\n")

    print("Average:")
    print(f"  Dice : {results_base['Dice'].mean():.2f}")
    print(f"  IoU  : {results_base['IoU'].mean():.2f}")
    print(f"  HD95 : {results_base['HD95'].mean():.2f}\n")

    for region in results:
        print(f"{region}:")
        print(f"  Dice : {results[region]['Dice'].mean():.2f}")
        print(f"  IoU  : {results[region]['IoU'].mean():.2f}")
        print(f"  HD95 : {results[region]['HD95'].mean():.2f}\n")

    save_path = config.save_path

    if best_performance < results_base['IoU'].mean():
        best_performance = results_base['IoU'].mean()
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)
    
    return best_performance


def validation_loop_brats2024(model, best_performance, valdataloader, config : Config):

    model.eval()

    results = {
        "WT": {"Dice": [], "IoU": [], "HD95": []},
        "TC": {"Dice": [], "IoU": [], "HD95": []},
        "ET": {"Dice": [], "IoU": [], "HD95": []},
        "SNFH": {"Dice": [], "IoU": [], "HD95": []},
        "RC": {"Dice": [], "IoU": [], "HD95": []},
        "NETC": {"Dice": [], "IoU": [], "HD95": []},
    }

    results_base = {"Dice": [], "IoU": [], "HD95": []}

    with torch.no_grad():
        for image, mask in tqdm(valdataloader):
            image = image.cuda()
            mask = mask.cuda()

            def predictor(x):
                out = model(x)
                return out["final_output"]

            prediction = sliding_window_inference(
                image.float(),
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=predictor,
                overlap=0.5,
                mode="gaussian"
            )

            prediction = torch.argmax(prediction, dim=1)
            
            gt = mask.cpu().numpy()[0]
            pred = prediction.cpu().numpy()[0]

            netc_gt, netc_pred = (gt == 1), (pred == 1)
            snfh_gt, snfh_pred = (gt == 2), (pred == 2)
            et_gt, et_pred     = (gt == 3), (pred == 3)
            rc_gt, rc_pred     = (gt == 4), (pred == 4)

            tc_gt = np.logical_or(netc_gt, et_gt)
            tc_pred = np.logical_or(netc_pred, et_pred)
            
            wt_gt = np.logical_or(tc_gt, snfh_gt)
            wt_pred = np.logical_or(tc_pred, snfh_pred)

            region_map = {
                "WT": (wt_pred, wt_gt),
                "TC": (tc_pred, tc_gt),
                "ET": (et_pred, et_gt),
                "SNFH": (snfh_pred, snfh_gt),
                "RC": (rc_pred, rc_gt),
                "NETC": (netc_pred, netc_gt),
            }

            case_sub_dice = []
            case_sub_iou = []
            case_sub_hd = []

            for region, (pred_r, gt_r) in region_map.items():
                dice = compute_safe_dice(pred_r, gt_r)
                iou  = compute_safe_iou(pred_r, gt_r)
                hd   = compute_safe_hd95(pred_r, gt_r)

                results[region]["Dice"].append(dice * 100)
                results[region]["IoU"].append(iou * 100)
                results[region]["HD95"].append(hd)

                if region in ["SNFH", "RC", "NETC", "ET"]:
                    case_sub_dice.append(dice)
                    case_sub_iou.append(iou)
                    case_sub_hd.append(hd)

            results_base["Dice"].append(np.mean(case_sub_dice) * 100)
            results_base["IoU"].append(np.mean(case_sub_iou) * 100)
            results_base["HD95"].append(np.mean(case_sub_hd))

    for region in results:
        for metric_name in results[region]:
            results[region][metric_name] = np.array(results[region][metric_name])

    for metric_name in results_base:
        results_base[metric_name] = np.array(results_base[metric_name])

    print(f"\nResults:\n")
    print("Average:")
    print(f"  Dice : {results_base['Dice'].mean():.2f}")
    print(f"  IoU  : {results_base['IoU'].mean():.2f}")
    print(f"  HD95 : {results_base['HD95'].mean():.2f}\n")

    for region in results:
        print(f"{region}:")
        print(f"  Dice : {results[region]['Dice'].mean():.2f}")
        print(f"  IoU  : {results[region]['IoU'].mean():.2f}")
        print(f"  HD95 : {results[region]['HD95'].mean():.2f}\n")

    save_path = config.save_path

    if best_performance < results_base['IoU'].mean():
        best_performance = results_base['IoU'].mean()
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)
    
    return best_performance


def validation_loop_binary3D(model, best_performance, valdataloader, config : Config):

    def get_largest_cc(segmentation):
        labels = label(segmentation) 
        if labels.max() == 0:
            return segmentation
        largest_cc = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
        return largest_cc.astype(np.uint8)

    model.eval()

    stride_xy  = config.stride_xy
    stride_z   = config.stride_z
    patch_size = config.patch_size  

    running_dice = []
    running_jaccard = []
    running_hd95 = []
    running_asd = []

    with torch.no_grad():
        for image, mask in tqdm(valdataloader):

            image_np = image[0, 0].cpu().numpy()
            mask_np  = mask[0].cpu().numpy()

            d, w, h = image_np.shape

            add_pad = False
            d_pad = w_pad = h_pad = 0

            if d < patch_size[0]:
                d_pad = patch_size[0] - d
                add_pad = True
            if w < patch_size[1]:
                w_pad = patch_size[1] - w
                add_pad = True
            if h < patch_size[2]:
                h_pad = patch_size[2] - h
                add_pad = True

            dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
            wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
            hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2

            if add_pad:
                image_np = np.pad(
                    image_np,
                    [(dl_pad, dr_pad), (wl_pad, wr_pad), (hl_pad, hr_pad)],
                    mode="constant",
                    constant_values=0,
                )

            dd, ww, hh = image_np.shape

            sd = math.ceil((dd - patch_size[0]) / stride_z)  + 1
            sw = math.ceil((ww - patch_size[1]) / stride_xy) + 1
            sh = math.ceil((hh - patch_size[2]) / stride_xy) + 1

            score_map = np.zeros((2, dd, ww, hh), dtype=np.float32)
            cnt       = np.zeros((dd, ww, hh), dtype=np.float32)

            for z in range(sd):
                zs = min(stride_z * z, dd - patch_size[0])
                for x in range(sw):
                    xs = min(stride_xy * x, ww - patch_size[1])
                    for y in range(sh):
                        ys = min(stride_xy * y, hh - patch_size[2])

                        patch = image_np[
                            zs:zs + patch_size[0],
                            xs:xs + patch_size[1],
                            ys:ys + patch_size[2],
                        ]

                        patch = (
                            torch.from_numpy(patch)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                            .cuda()
                        )

                        output = model(patch)
                        if isinstance(output, dict):
                            output = output["final_output"]

                        output = torch.softmax(output, dim=1)
                        prob_fg = output[0, 1].cpu().numpy()

                        score_map[
                            1,
                            zs:zs + patch_size[0],
                            xs:xs + patch_size[1],
                            ys:ys + patch_size[2],
                        ] += prob_fg

                        cnt[
                            zs:zs + patch_size[0],
                            xs:xs + patch_size[1],
                            ys:ys + patch_size[2],
                        ] += 1

            score_map[1] /= cnt

            prediction = (score_map[1] > 0.5).astype(np.uint8)

            if add_pad:
                prediction = prediction[
                    dl_pad:dl_pad + d,
                    wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h,
                ]

            prediction = get_largest_cc(prediction)

            if np.any(prediction) and np.any(mask_np):
                running_dice.append(metric.binary.dc(prediction, mask_np))
                running_jaccard.append(metric.binary.jc(prediction, mask_np))
                running_hd95.append(metric.binary.hd95(prediction, mask_np))
                running_asd.append(metric.binary.asd(prediction, mask_np))
            else:
                running_dice.append(0.0)
                running_jaccard.append(0.0)
                running_hd95.append(0.0)
                running_asd.append(0.0)

    def safe_mean(x):
        x = np.asarray(x)
        return float("nan") if x.size == 0 else x.mean()

    mean_dice    = safe_mean(running_dice) * 100
    mean_jaccard = safe_mean(running_jaccard) * 100
    mean_hd      = safe_mean(running_hd95)
    mean_asd     = safe_mean(running_asd)

    print("Validation Step Complete on Model!\n")
    print(f"Dice Score: {mean_dice} (Best Dice : {best_performance})")
    print(f"Jaccard Index: {mean_jaccard}")
    print(f"HD95: {mean_hd}")
    print(f"ASD: {mean_asd}")

    save_path = config.save_path

    if best_performance < mean_dice:
        best_performance = mean_dice
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)

    return best_performance


if __name__ == "__main__":

    config = Config()
    test_dataset = LAHeart(base_dir=config.root_dir, transforms=None, config=config, is_training=False)
    validation_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=1, num_workers=config.num_workers, shuffle=False)
    model = VNet(n_channels=config.num_modalities, n_classes=config.num_classes).cuda()
    model.load_state_dict(torch.load("./Results/LA_8/best_model.pth", map_location='cuda'))
    best_performance = 100.0
    best_performance = validation_loop_binary3D(model, best_performance, validation_dataloader, config)