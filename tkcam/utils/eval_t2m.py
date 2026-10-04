import os

import clip
import numpy as np
import torch
# from scipy import linalg
from tkcam.utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from tkcam.utils.motion_process import recover_from_ric
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe

@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0
    l1_dist = l1_dist / num_poses if num_poses > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0
    l1_dist = l1_dist / num_poses if num_poses > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        # Only exclude CLIP weights if NOT fine-tuning
        if not trans.finetune_clip:
            clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # Detect id_embedding mode from the model's conditioning_mode attribute so we
    # are never misled by the eval_val_loader returning the standard 7-element
    # text-dataset batch format even when the model was trained with sample IDs.
    is_id_embedding_mode = (
        getattr(trans, 'conditioning_mode', 'clip') == 'id_embedding'
    )
    last_batch_data = {}  # Store last batch for visualization

    # for i in range(1):
    for batch in val_loader:
        # Check if this is id_embedding mode (3 elements) or text mode (7 elements)
        if len(batch) == 3:
            is_id_embedding_mode = True
            sample_idx, pose, m_length = batch
            sample_idx = sample_idx.cuda()
            pose = pose.cuda()
            m_length = m_length.cuda()
            bs, seq = pose.shape[:2]
            
            # Generate motions using sample indices as conditions
            mids = trans.generate(sample_idx, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale, temperature=1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)
            
            # Store for visualization
            last_batch_data = {
                'pred_motions': pred_motions,
                'pose': pose,
                'm_length': m_length,
                'sample_idx': sample_idx,
                'bs': bs
            }
            
            # Skip text-based metric computation for id_embedding mode
            nb_sample += bs
            continue
            
        # Standard text-based evaluation
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]

        # When the model uses id_embedding mode but the eval_val_loader returns
        # the standard 7-element text batch, skip generation — passing string
        # text to an id_embedding model would crash with AttributeError.
        if is_id_embedding_mode:
            nb_sample += bs
            continue

        # (b, seqlen)
        mids = trans.generate(clip_text, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)
        
        # Store for visualization
        last_batch_data = {
            'pred_motions': pred_motions,
            'pose': pose,
            'm_length': m_length,
            'clip_text': clip_text,
            'bs': bs
        }

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Animation generation — runs BEFORE metric guards so it works in id_embedding
    # mode even when motion_annotation_list is empty (no text-metric batches).
    if save_anim and nb_sample > 0 and last_batch_data:
        # Extract data from last_batch_data
        pred_motions = last_batch_data['pred_motions']
        pose = last_batch_data['pose']
        m_length = last_batch_data['m_length']
        bs = last_batch_data['bs']

        # Use min(3, bs) to handle small batches
        num_viz = min(3, bs)
        rand_idx = torch.randint(bs, (num_viz,))

        # Save PREDICTED trajectories
        pred_data = pred_motions[rand_idx].detach().cpu().numpy()

        # Handle captions based on mode
        if is_id_embedding_mode:
            sample_idx = last_batch_data['sample_idx']
            captions = [f"Sample ID: {sample_idx[k].item()}" for k in rand_idx]
        else:
            clip_text = last_batch_data['clip_text']
            captions = [clip_text[k] for k in rand_idx]

        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(pred_data, save_dir, captions, lengths)

        # Save GROUND TRUTH trajectories for comparison
        gt_data = pose[rand_idx].detach().cpu().numpy()
        if is_id_embedding_mode:
            gt_captions = [f"[GT] Sample ID: {sample_idx[k].item()}" for k in rand_idx]
        else:
            gt_captions = [f"[GT] {clip_text[k]}" for k in rand_idx]
        save_dir_gt = os.path.join(out_dir, 'animation', 'E%04d_groundtruth' % ep)
        os.makedirs(save_dir_gt, exist_ok=True)
        plot_func(gt_data, save_dir_gt, gt_captions, lengths)

        print(f"Saved validation animations: predicted (E{ep:04d}) + ground truth (E{ep:04d}_groundtruth)")
    elif save_anim:
        print(f"Warning: Skipping animation generation for epoch {ep} - no samples processed")

    # When the model uses id_embedding mode and the eval_val_loader returns
    # standard 7-element text batches, every batch is skipped — motion lists
    # stay empty.  Computing FID/diversity on dummy zeros would poison best_fid
    # to 0.0 on epoch 0, preventing net_best_fid.tar from ever being saved.
    # Guard: skip metric computation and return the incoming best values intact.
    if not motion_annotation_list or not motion_pred_list:
        print(f"[Eval Ep {ep}] No text-metric samples processed (id_embedding mode). "
              "Skipping FID/diversity/R-precision — returning current best values.")
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

    # Compute metrics only for text-based evaluation
    if not is_id_embedding_mode:
        # Safety check for empty motion lists (single sample debugging)
        if motion_annotation_list and motion_pred_list:
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        else:
            print("Warning: No motion samples for evaluation, using dummy data")
            motion_annotation_np = motion_pred_np = np.zeros((1, 512))
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)

        diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
        R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

        matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
        matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        print(msg)
    else:
        # Set all metrics to 0 for id_embedding mode
        fid = 0.0
        diversity_real = 0.0
        diversity = 0.0
        R_precision_real = np.array([0.0, 0.0, 0.0])
        R_precision = np.array([0.0, 0.0, 0.0])
        matching_score_real = 0.0
        matching_score_pred = 0.0
        print(f"--> \t Eva. Ep {ep} (id_embedding mode - {nb_sample} samples generated, text metrics skipped)")

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        # Only exclude CLIP weights if NOT fine-tuning
        if not trans.finetune_clip:
            clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    last_batch_data = {}  # Store last batch for visualization
    # Detect id_embedding mode from the model's conditioning_mode attribute so we
    # are never misled by the eval_val_loader returning the standard 7-element
    # text-dataset batch format even when the model was trained with sample IDs.
    is_id_embedding_mode = (
        getattr(trans, 'conditioning_mode', 'clip') == 'id_embedding'
    )
    # for i in range(1):
    for batch in val_loader:
        # id_embedding dataset yields (sample_idx, pose, m_length)
        if len(batch) == 3:
            is_id_embedding_mode = True
            sample_idx, pose, m_length = batch
            sample_idx = sample_idx.cuda()
            pose = pose.cuda().float()
            m_length = m_length.cuda().long()
            bs, seq = pose.shape[:2]

            code_indices, all_codes = vq_model.encode(pose)
            if ep == 0:
                pred_ids = code_indices[..., 0:1]
            else:
                pred_ids = trans.generate(
                    code_indices[..., 0], sample_idx,
                    torch.div(m_length, 4, rounding_mode='floor'),
                    temperature=temperature, cond_scale=cond_scale)

            pred_motions_id = vq_model.forward_decoder(pred_ids)
            # Store for visualization
            last_batch_data = {
                'pred_motions': pred_motions_id,
                'pose': pose,
                'm_length': m_length,
                'sample_idx': sample_idx,
                'bs': bs,
            }

            nb_sample += bs
            continue

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]

        # When the model uses id_embedding mode but the eval_val_loader returns
        # the standard 7-element text batch, skip generation — passing string
        # text to an id_embedding model would crash with AttributeError.
        if is_id_embedding_mode:
            nb_sample += bs
            continue

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'),
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'), force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)
        # Store for visualization (captures the last text-mode batch)
        last_batch_data = {
            'pred_motions': pred_motions,
            'pose': pose,
            'm_length': m_length,
            'clip_text': clip_text,
            'bs': bs,
        }

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Animation generation — runs BEFORE metric guards so it works in id_embedding
    # mode even when motion_annotation_list is empty (no text-metric batches).
    if save_anim and nb_sample > 0 and last_batch_data:
        pred_motions_viz = last_batch_data['pred_motions']
        pose_viz = last_batch_data['pose']
        m_length_viz = last_batch_data['m_length']
        bs_viz = last_batch_data['bs']

        num_viz = min(3, bs_viz)
        rand_idx = torch.randint(bs_viz, (num_viz,))

        # Save PREDICTED trajectories
        pred_data = pred_motions_viz[rand_idx].detach().cpu().numpy()

        # Handle captions based on mode
        if is_id_embedding_mode:
            sample_idx_viz = last_batch_data['sample_idx']
            captions = [f"Sample ID: {sample_idx_viz[k].item()}" for k in rand_idx]
        else:
            clip_text_viz = last_batch_data['clip_text']
            captions = [clip_text_viz[k] for k in rand_idx]

        lengths = m_length_viz[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(pred_data, save_dir, captions, lengths)

        # Save GROUND TRUTH trajectories for comparison
        gt_data = pose_viz[rand_idx].detach().cpu().numpy()
        if is_id_embedding_mode:
            gt_captions = [f"[GT] Sample ID: {sample_idx_viz[k].item()}" for k in rand_idx]
        else:
            gt_captions = [f"[GT] {clip_text_viz[k]}" for k in rand_idx]
        save_dir_gt = os.path.join(out_dir, 'animation', 'E%04d_groundtruth' % ep)
        os.makedirs(save_dir_gt, exist_ok=True)
        plot_func(gt_data, save_dir_gt, gt_captions, lengths)

        print(f"Saved validation animations: predicted (E{ep:04d}) + ground truth (E{ep:04d}_groundtruth)")
    elif save_anim:
        print(f"Warning: Skipping animation generation for epoch {ep} - no samples processed")

    # When the model uses id_embedding mode and the eval_val_loader returns
    # standard 7-element text batches, every batch is skipped — motion lists
    # stay empty.  Computing FID/diversity on dummy zeros would poison best_fid
    # to 0.0 on epoch 0, preventing net_best_fid.tar from ever being saved.
    # Guard: skip metric computation and return the incoming best values intact.
    if not motion_annotation_list or not motion_pred_list:
        print(f"[Eval Ep {ep}] No text-metric samples processed (id_embedding mode). "
              "Skipping FID/diversity/R-precision — returning current best values.")
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(val_loader, vq_model, trans, repeat_id, eval_wrapper, num_joint,
                                       cond_scale=2, temperature=1, topkr=0.9, cal_l1=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'), topk_filter_thres=topkr,
                                  temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'), force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0
    l1_dist = l1_dist / num_poses if num_poses > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, clip_text, torch.div(m_length, 4, rounding_mode='floor'), temperature=1, cond_scale=res_cond_scale)
                # pred_codes = trans(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'), force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, torch.div(m_length, 4, rounding_mode='floor'), time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, clip_text, torch.div(m_length, 4, rounding_mode='floor'), temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, torch.div(m_length, 4, rounding_mode='floor'), force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # Safety check for empty motion lists (single sample debugging)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    else:
        print("Warning: No motion samples for evaluation, using dummy data")
        motion_annotation_np = motion_pred_np = np.zeros((1, 512))
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])
    R_precision = R_precision / nb_sample if nb_sample > 0 else np.array([0.0, 0.0, 0.0])

    matching_score_real = matching_score_real / nb_sample if nb_sample > 0 else 0.0
    matching_score_pred = matching_score_pred / nb_sample if nb_sample > 0 else 0.0

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality