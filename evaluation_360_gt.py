
#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
import tools.transformations as tr
from tools.pose_evaluation_utils import quat_pose_to_mat
from scipy.spatial.transform import Rotation
# import evo.main_ape as ape
from kiss_icp.metrics import absolute_trajectory_error, sequence_error
def takeSort(elem):
    return int(elem[3:])

class kittiOdomEval():
    def __init__(self, config):
        self.lengths = [100,200,300,400,500,600,700,800]
        self.num_lengths = len(self.lengths)
        self.gt_dir     = config.gt_dir
        self.pose_dir  = config.pose_dir
        self.dataset_dir = config.dataset_dir
        self.eval_seqs  = []
        self.pose_format = config.pose_format
        # gt_files = glob.glob(config.gt_dir + '/*.txt')
        # gt_files = [os.path.split(f)[1] for f in gt_files]
        # self.seqs_with_gt = [os.path.splitext(f)[0] for f in gt_files]
        
        # evalute all files in the folder
        if config.eva_seqs == '*':
            if not os.path.exists(self.dataset_dir):
                print('File path error!')
                exit()
            for name in os.listdir(self.dataset_dir):
                # 获取name前三位字符

                if (name[0:3] != "seq"): continue

                if os.path.isdir(os.path.join(self.dataset_dir, name)):
                    self.eval_seqs.append(str(name))
            # 排序eval_seqs
            self.eval_seqs.sort(key=takeSort)
        else:
            seqs = config.eva_seqs.split(',')
            self.eval_seqs = [str(s) for s in seqs]


        # # Ref: https://github.com/MichaelGrupp/evo/wiki/Plotting
        # os.system("evo_config set plot_seaborn_style whitegrid \
        #             plot_linewidth 1.0 \
        #             plot_fontfamily sans-serif \
        #             plot_fontscale 1.0 \
        #             plot_figsize 10 10 \
        #             plot_export_format pdf")

    def toCameraCoord(self, pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        Tr = np.array([[0.99992906, 0.0057743, 0.01041756, 0.77104934],
                 [0.00580536, -0.99997879, -0.00295331, 0.29854144],
                 [0.01040029, 0.00301357, -0.99994137, -0.83628022],
                 [0, 0, 0, 1]])
        Tr_inv = np.linalg.inv(Tr)            
        rot = np.matmul(Tr_inv, np.matmul(pose_mat, Tr))
        return rot 

    def loadPoses(self, file_name, toCameraCoord, format, seq_num):
        '''
            Each line in the file should follow one of the following structures
            time x y z rx ry rz rw
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            if(format == 'tum'):
                # xyz
                P[0, 3] = line_split[1]
                P[1, 3] = line_split[2]
                P[2, 3] = line_split[3]
                # Q to R
                q = np.array([line_split[4],line_split[5],line_split[6],line_split[7]]) #xyzw
                r = Rotation.from_quat(q)
                rot_mat = r.as_matrix()
                P[0:3, 0:3] = rot_mat
                if seq_num == "02":
                    frame_idx = 4391 + cnt
                else:
                    frame_idx = cnt
            elif(format == 'kitti'):
                withIdx = int(len(line_split)==13)
                for row in range(3):
                    for col in range(4):
                        P[row, col] = line_split[row*4 + col + withIdx]
                if withIdx:
                    frame_idx = line_split[0]
                else:
                    frame_idx = cnt
            else :
                print('Unknown pose format!')
                exit()
            if toCameraCoord:
                poses[frame_idx] = self.toCameraCoord(P)
            else:
                poses[frame_idx] = P
        return poses
    
    def loadgtPoses(self, file_name, toCameraCoord, pose_ref, seq_num):
        '''
            Each line in the file should follow one of the following structures
            time x y z rx ry rz rw
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        gt_t_p = np.eye(4)
        untrans = True
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]

            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = int(line_split[0])
            else:
                frame_idx = cnt
            if (seq_num == "02") and (frame_idx < 4391):
                continue
            if toCameraCoord:
                poses[frame_idx] = self.toCameraCoord(P)
            else:
                poses[frame_idx] = P
            if untrans:
                gt_t_p = np.matmul(pose_ref[frame_idx], np.linalg.inv(poses[frame_idx]))
                poses[frame_idx] = np.matmul(gt_t_p, poses[frame_idx])
                untrans = False
            else:
                poses[frame_idx] = np.matmul(gt_t_p, poses[frame_idx])
        return poses

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        # dist = [0]
        sort_frame_idx = sorted(poses.keys())
        dist = {sort_frame_idx[0]: 0}
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0,3] - P2[0,3]
            dy = P1[1,3] - P2[1,3]
            dz = P1[2,3] - P2[2,3]
            dist[sort_frame_idx[i+1]] = dist[sort_frame_idx[i]]+np.sqrt(dx**2+dy**2+dz**2)
        self.distance = dist[sort_frame_idx[-1]]
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d,1.0),-1.0))

    def translationError(self, pose_error):
        dx = pose_error[0,3]
        dy = pose_error[1,3]
        dz = pose_error[2,3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_, pose_gt_num):
        dist_num = sorted(dist.keys())
        for i in range(pose_gt_num, len(dist_num), 1):
            dist_num_i = dist_num[i]
            if dist[dist_num_i] > (dist[first_frame] + len_):
                return dist_num_i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        self.max_speed = 0
        # pre-compute distances (from ground truth as reference)
        dist = self.trajectoryDistances(poses_gt)
        # every second, kitti data 10Hz
        # self.step_size = 10
        # for all start positions do
        # for first_frame in range(9, len(poses_gt), self.step_size):
        sort_frame_idx = sorted(poses_gt.keys())
        last_fist_frame = -10
        for pose_gt_num in range(0, len(poses_gt)):
            first_frame = sort_frame_idx[pose_gt_num]
            if first_frame - last_fist_frame < 10:
                continue
            last_fist_frame = first_frame
            # for all segment lengths do
            for i in range(self.num_lengths):
                # current length
                len_ = self.lengths[i]
                # compute last frame of the segment
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_, pose_gt_num)

                # Continue if sequence not long enough
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # compute rotational and translational errors, relative pose error (RPE)
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # compute speed 
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1*num_frames)   # 10Hz
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name,'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err 

    def plot_xyz(self, seq, poses_ref, poses_pred, plot_path_dir):
        
        def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
            """
                plot a path/trajectory based on xyz coordinates into an axis
                :param axarr: an axis array (for x, y & z) e.g. from 'fig, axarr = plt.subplots(3)'
                :param traj: trajectory
                :param style: matplotlib line style
                :param color: matplotlib color
                :param label: label (for legend)
                :param alpha: alpha value for transparency
            """
            x = range(0, len(positions_xyz))
            xlabel = "index"
            ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
            # plt.title('PRY')
            for i in range(0, 3):
                axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('XYZ')           

        fig, axarr = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))  
        
        pred_xyz = np.array([p[:3, 3] for _,p in poses_pred.items()])
        traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)
        if poses_ref:
            ref_xyz = np.array([p[:3, 3] for _,p in poses_ref.items()])
            traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)
      
        name = "{}_xyz".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)       
        # plt.show()
        pdf.close()

    def plot_rpy(self, seq, poses_ref, poses_pred, plot_path_dir, axes='szxy'):
        
        def traj_rpy(axarr, orientations_euler, style='-', color='black', title="", label="", alpha=1.0):
            """
            plot a path/trajectory's Euler RPY angles into an axis
            :param axarr: an axis array (for R, P & Y) e.g. from 'fig, axarr = plt.subplots(3)'
            :param traj: trajectory
            :param style: matplotlib line style
            :param color: matplotlib color
            :param label: label (for legend)
            :param alpha: alpha value for transparency
            """
            x = range(0, len(orientations_euler))
            xlabel = "index"
            ylabels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
            # plt.title('PRY')
            for i in range(0, 3):
                axarr[i].plot(x, np.rad2deg(orientations_euler[:, i]), style,
                            color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('PRY')           

        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))

        pred_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_pred.items()])
        traj_rpy(axarr_rpy, pred_rpy, '-', 'b', title='RPY', label='Ours', alpha=1.0)
        if poses_ref:
            ref_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_ref.items()])
            traj_rpy(axarr_rpy, ref_rpy, '-', 'r', label='GT', alpha=1.0)

        name = "{}_rpy".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig_rpy.tight_layout()
        pdf.savefig(fig_rpy)       
        # plt.show()
        pdf.close()

    def plotPath_2D_3(self, seq, poses_gt, poses_result, plot_path_dir):
        '''
            plot path in XY, XZ and YZ plane
        '''
        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        ### get the value
        if poses_gt: 
            poses_gt = [(k,poses_gt[k]) for k in sorted(poses_gt.keys())]
            x_gt = np.asarray([pose[0,3] for _,pose in poses_gt])
            y_gt = np.asarray([pose[1,3] for _,pose in poses_gt])
            z_gt = np.asarray([pose[2,3] for _,pose in poses_gt])
        poses_result = [(k,poses_result[k]) for k in sorted(poses_result.keys())]
        x_pred = np.asarray([pose[0,3] for _,pose in poses_result])
        y_pred = np.asarray([pose[1,3] for _,pose in poses_result])
        z_pred = np.asarray([pose[2,3] for _,pose in poses_result])
        
        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        plt.subplot(1,3,1)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
                            for lims, mean_ in ((xlim, xmean),
                                                (ylim, ymean))
                            for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,2)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,3)
        ax = plt.gca()
        if poses_gt: plt.plot(y_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(y_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        png_title = "{}_path".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        # plt.show()
        plt.close()

    def plotPath_3D(self, seq, poses_gt, poses_result, plot_path_dir):
        """
            plot the path in 3D space
        """
        from mpl_toolkits.mplot3d import Axes3D

        start_point = [[0], [0], [0]]
        fontsize_ = 8
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        poses_dict = {}      
        poses_dict["Ours"] = poses_result
        if poses_gt:
            poses_dict["Ground Truth"] = poses_gt

        fig = plt.figure(figsize=(8,8), dpi=110)
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')

        for key,_ in poses_dict.items():
            plane_point = []
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                plane_point.append([pose[0,3], pose[2,3], pose[1,3]])
            plane_point = np.asarray(plane_point)
            style = style_pred if key == 'Ours' else style_gt
            plt.plot(plane_point[:,0], plane_point[:,1], plane_point[:,2], style, label=key)  
        plt.plot(start_point[0], start_point[1], start_point[2], style_O, label='Start Point')

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)
        plot_radius = max([abs(lim - mean_)
                        for lims, mean_ in ((xlim, xmean),
                                            (ylim, ymean),
                                            (zlim, zmean))
                        for lim in lims])
        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

        ax.legend()
        # plt.legend(loc="upper right", prop={'size':fontsize_}) 
        ax.set_xlabel('x (m)', fontsize=fontsize_)
        ax.set_ylabel('z (m)', fontsize=fontsize_)
        ax.set_zlabel('y (m)', fontsize=fontsize_)
        ax.view_init(elev=20., azim=-35)

        png_title = "{}_path_3D".format(seq)
        plt.savefig(plot_path_dir+"/"+png_title+".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        # plt.show()
        plt.close()

    def plotError_segment(self, seq, avg_segment_errs, plot_error_dir):
        '''
            avg_segment_errs: dict [100: err, 200: err...]
        '''
        fontsize_ = 15
        plot_y_t = []
        plot_y_r = []
        plot_x = []
        for idx, value in avg_segment_errs.items():
            if value == []:
                continue
            plot_x.append(idx)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Translation Error (%)',fontsize=fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize=fontsize_)
        png_title = "{}_error_seg".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    def plotError_speed(self, seq, avg_speed_errs, plot_error_dir):
        '''
            avg_speed_errs: dict [s1: err, s2: err...]
        '''
        fontsize_ = 15
        plot_y_t = []
        plot_y_r = []
        plot_x = []
        for idx, value in avg_speed_errs.items():
            if value == []:
                continue
            plot_x.append(idx * 3.6)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)        
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Translation Error (%)',fontsize = fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize = fontsize_)
        png_title = "{}_error_speed".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    def computeSegmentErr(self, seq_errs):
        '''
            This function calculates average errors for different segment.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_  = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def computeSpeedErr(self, seq_errs):
        '''
            This function calculates average errors for different speed.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for s in range(2, 25, 2):
            segment_errs[s] = []

        # Get errors
        for err in seq_errs:
            speed = err[4]
            t_err = err[2]
            r_err = err[1]
            for key in segment_errs.keys():
                if np.abs(speed - key) < 2.0:
                    segment_errs[key].append([t_err, r_err])

        # Compute average
        for key in segment_errs.keys():
            if segment_errs[key] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[key])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[key])[:,1])
                avg_segment_errs[key] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[key] = []
        return avg_segment_errs

    def call_evo_traj(self, pred_file, save_file, gt_file=None, plot_plane='xy'):
        command = ''
        if os.path.exists(save_file): os.remove(save_file)
        
        if gt_file != None:
            command = ("evo_traj kitti %s --ref=%s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, gt_file, plot_plane, save_file)
        else:
            command = ("evo_traj kitti %s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, plot_plane, save_file)
        os.system(command)

    def eval(self, toCameraCoord):
        '''
            to_camera_coord: whether the predicted pose needs to be convert to camera coordinate
        '''
        # eval_dir = self.result_dir
        # if not os.path.exists(eval_dir): os.makedirs(eval_dir)

        total_err = []
        ave_errs = {}
        ate_errs = {}       
        for seq in self.eval_seqs:
            seq_num = int(seq[3:])
            if(seq_num < 10):
                seq_num = '0' + str(seq_num)
            else:
                seq_num = str(seq_num)
            eva_seq_dir = os.path.join(self.dataset_dir, seq)
            if not os.path.exists(eva_seq_dir): 
                print("Dir %s couldn't open!"%(eva_seq_dir))
                exit() 
            pred_file_name = os.path.join(eva_seq_dir, self.pose_dir)
            # pred_file_name = self.result_dir + '/{}.txt'.format(seq)
            gt_file_name = os.path.join("/media/oliver/Elements SE/dataset/kitti_360/data_poses/2013_05_28_drive_00" + seq_num + "_sync", self.gt_dir)
            #gt_file_name   = self.gt_dir + '/{}.txt'.format(seq)
            save_file_name = eva_seq_dir + '/{}.pdf'.format(seq)
            assert os.path.exists(pred_file_name), "File path error: {}".format(pred_file_name)
            # assert os.path.exists(gt_file_name), "File path error: {}".format(gt_file_name)
            
            # ----------------------------------------------------------------------
            # load pose
            # if seq in self.seqs_with_gt:
            #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=gt_file_name)
            # else:
            #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=None)
            #     continue
            
            poses_result = self.loadPoses(pred_file_name, toCameraCoord=False, format = self.pose_format, seq_num = seq_num)
            # if not os.path.exists(eva_seq_dir): os.makedirs(eva_seq_dir) 

            if not os.path.exists(gt_file_name):
                self.calcSequenceErrors(poses_result, poses_result)
                print ("\nSequence: " + str(seq))
                print ('Distance (m): %d' % self.distance)
                print ('Max speed (km/h): %d' % (self.max_speed*3.6))
                self.plot_rpy(seq, None, poses_result, eva_seq_dir)
                self.plot_xyz(seq, None, poses_result, eva_seq_dir)
                self.plotPath_3D(seq, None, poses_result, eva_seq_dir)
                self.plotPath_2D_3(seq, None, poses_result, eva_seq_dir)
                continue
          
            poses_gt = self.loadgtPoses(gt_file_name, toCameraCoord=True, pose_ref=poses_result, seq_num = seq_num)

            # ----------------------------------------------------------------------
            # compute sequence errors
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, eva_seq_dir + '/{}_error.txt'.format(seq))

            total_err += seq_err

            # ----------------------------------------------------------------------
            # Compute segment errors
            avg_segment_errs = self.computeSegmentErr(seq_err)
            avg_speed_errs   = self.computeSpeedErr(seq_err)

            # ----------------------------------------------------------------------
            # compute overall error
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)

            # ----------------------------------------------------------------------
            # compute absolute error
            ate_rot, ate_trans = absolute_trajectory_error(list(poses_gt.values()), list(poses_result.values()))

            print ("\nSequence: " + str(seq))
            print ('Distance (m): %d' % self.distance)
            print ('Max speed (km/h): %d' % (self.max_speed*3.6))
            print ("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
            print ("Average sequence rotational error (deg/m): {0:.4f}".format(ave_r_err/np.pi * 180))
            print ("Absoulte Trajectory Error (ATE) (m): {0:.4f}".format(ate_trans))
            print ("Absoulte Rotational Error (ARE) (rad): {0:.4f}\n".format(ate_rot))
            with open(eva_seq_dir + '/%s_stats.txt' % seq, 'w') as f:
                f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
                f.writelines('Average sequence rotation error (deg/m):  {0:.4f}\n'.format(ave_r_err/np.pi * 180))
                f.writelines("Absoulte Trajectory Error (ATE) (m): {0:.4f}\n".format(ate_trans))
                f.writelines("Absoulte Rotational Error (ARE) (rad): {0:.4f}\n".format(ate_rot))
            ave_errs[seq] = [ave_t_err, ave_r_err]
            ate_errs[seq] = [ate_trans, ate_rot]

            # ----------------------------------------------------------------------
            # Ploting
            self.plot_rpy(seq, poses_gt, poses_result, eva_seq_dir)
            self.plot_xyz(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_3D(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_2D_3(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotError_segment(seq, avg_segment_errs, eva_seq_dir)
            self.plotError_speed(seq, avg_speed_errs, eva_seq_dir)

            plt.close('all')

            # ----------------------------------------------------------------------
            # evo ape compute
            # parser = ape.parser()
            # args = parser.parse_args([self.pose_format, gt_file_name, pred_file_name, "-va", "--logfile", eva_seq_dir + "/evo_ape.log", "--plot_mode", "xy", "--save_plot", eva_seq_dir + "/evo_ape"])
            # ape.run(args)


        total_avg_segment_errs = self.computeSegmentErr(total_err)
        total_avg_speed_errs   = self.computeSpeedErr(total_err)
        # compute overall error
        ave_t_err, ave_r_err = self.computeOverallErr(total_err)
        # compute mean error
        mean_t_err = 0
        mean_r_err = 0
        for seq, ave_err in ave_errs.items():
            mean_t_err += ave_err[0]
            mean_r_err += ave_err[1]
        mean_t_err /= len(ave_errs)
        mean_r_err /= len(ave_errs)
        mean_at_err = 0
        mean_ar_err = 0
        for seq, ate_err in ate_errs.items():
            mean_at_err += ate_err[0]
            mean_ar_err += ate_err[1]
        mean_at_err /= len(ate_errs)
        mean_ar_err /= len(ate_errs)
        print ("\nSequence All: ")
        print ("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
        print ("Average sequence rotational error (deg/m): {0:.4f}".format(ave_r_err/np.pi * 180))
        print ("Mean sequence translational RMSE (%):   {0:.4f}".format(mean_t_err * 100))
        print ("Mean sequence rotational error (deg/m): {0:.4f}".format(mean_r_err/np.pi * 180))
        print ("Mean Absoulte Trajectory Error (ATE) (m): {0:.4f}".format(mean_at_err))
        print ("Mean Absoulte Rotational Error (ARE) (rad): {0:.4f}\n".format(mean_ar_err))
        with open(self.dataset_dir + '/total_stats.txt', 'w') as f:
            f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
            f.writelines('Average sequence rotation error (deg/m):  {0:.4f}\n'.format(ave_r_err/np.pi * 180))
            f.writelines('Mean sequence translation RMSE (%):    {0:.4f}\n'.format(mean_t_err * 100))
            f.writelines('Mean sequence rotation error (deg/m):  {0:.4f}\n'.format(mean_r_err/np.pi * 180))
            f.writelines("Mean Absoulte Trajectory Error (ATE) (m): {0:.4f}\n".format(mean_at_err))
            f.writelines("Mean Absoulte Rotational Error (ARE) (rad): {0:.4f}\n".format(mean_ar_err))
            f.writelines('\n\n')
            for seq, ave_err in ave_errs.items():
                f.writelines('%s:\n' % seq)
                f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_err[0] * 100))
                f.writelines('Average sequence rotation error (deg/m):  {0:.4f}\n'.format(ave_err[1]/np.pi * 180))
                f.writelines("Absoulte Trajectory Error (ATE) (m): {0:.4f}\n".format(ate_errs[seq][0]))
                f.writelines("Absoulte Rotational Error (ARE) (rad): {0:.4f}\n\n".format(ate_errs[seq][1]))


        # ----------------------------------------------------------------------
        # Ploting       
        self.plotError_segment('total_error_seg', total_avg_segment_errs, self.dataset_dir)
        self.plotError_speed('total_error_speed', total_avg_speed_errs, self.dataset_dir)


        # if ave_errs:
        #     with open(eval_dir + '/all_stats.txt', 'w') as f:
        #         for seq, ave_err in ave_errs.items():
        #             f.writelines('%s:\n' % seq)
        #             f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_err[0] * 100))
        #             f.writelines('Average sequence rotation error (deg/m):  {0:.4f}\n\n'.format(ave_err[1]/np.pi * 180))

            # parent_path, model_step = os.path.split(os.path.normpath(eval_dir))
            # with open(os.path.join(parent_path, 'test_statistics.txt'), 'a') as f:
            #     f.writelines('------------------ %s -----------------\n' % model_step)
            #     for seq, ave_err in ave_errs.items():
            #         f.writelines('%s:\n' % seq)
            #         f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_err[0] * 100))
            #         f.writelines('Average sequence rotation error (deg/m):  {0:.4f}\n\n'.format(ave_err[1]/np.pi * 180))
                      
        # print ("-------------------------------------------------")
        # for seq in range(len(ave_t_errs)):
        #     print ("{0:.2f}".format(ave_t_errs[seq]*100))
        #     print ("{0:.2f}".format(ave_r_errs[seq]/np.pi*180*100))
        # print ("-------------------------------------------------")

     
if __name__ == '__main__':
    # 获取当前文件的绝对路径
    file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    dir_path = os.path.dirname(os.path.dirname(file_path)) # '/home/oliver/catkin_ros2/src/kiss-icp/results
    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--dataset_dir',type=str, default=dir_path + '/230804_r_sem', help='Directory path of the testing dataset') # + '/230627_mul'
    parser.add_argument('--gt_dir',     type=str, default='poses.txt',  help='Filename of the ground truth odometry')
    parser.add_argument('--pose_dir',     type=str, default='path.txt',  help='Filename of evaluated odometry')
    parser.add_argument('--eva_seqs',   type=str, default='*',      help='The sequences to be evaluated, split by (,), or (*)')
    parser.add_argument('--pose_format',     type=str, default='tum',  help='Format of the pose file, kitti or tum')
    # parser.add_argument('--gt_dir',     type=str, default='./ground_truth_pose',  help='Directory path of the ground truth odometry')
    # parser.add_argument('--result_dir', type=str, default='./data/',              help='Directory path of storing the odometry results')
    # parser.add_argument('--eva_seqs',   type=str, default='09_pred,10_pred,11_pred',      help='The sequences to be evaluated') 
    parser.add_argument('--toCameraCoord',   type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    pose_eval = kittiOdomEval(args)
    pose_eval.eval(toCameraCoord=args.toCameraCoord)   # set the value according to the predicted results

    
