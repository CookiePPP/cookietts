import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, loss_terms, teacher_force_till, p_teacher_forcing, diagonality=None, avg_prob=None):
        if diagonality is not None:
            self.add_scalar("training.attention_alignment_diagonality", diagonality, iteration)
        if avg_prob is not None:
            self.add_scalar("training.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("training.p_teacher_forcing", p_teacher_forcing, iteration)
        self.add_scalar("training.teacher_force_till", teacher_force_till, iteration)
        self.add_scalar("duration", duration, iteration)
        
        if iteration%10 == 0:# log every 10th iter
            self.log_additional_losses(loss_terms, iteration, prepend='training.')
        
    def log_validation(self, reduced_loss, model, y, y_pred, iteration, loss_terms, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_scalar("validation.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("validation.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("validation.p_teacher_forcing", val_p_teacher_forcing, iteration)
        self.add_scalar("validation.teacher_force_till", val_teacher_force_till, iteration)
        
        self.log_additional_losses(loss_terms, iteration, prepend='validation.')
        
        _, _, _, alignments, *_ = y_pred
        
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot alignment, mel target and predicted, gate target and predicted
        idx = 0 # plot longest audio file
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        
        idx = 1 # and randomly pick a second one to plot
        self.add_image(
            "alignment2",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
    
    def log_infer(self, reduced_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("infer.loss", reduced_loss, iteration)
        self.add_scalar("infer.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("infer.average_max_attention_weight", avg_prob, iteration)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments, *_ = y_pred
        if mel_outputs_postnet is not None:
            mel_outputs = mel_outputs_postnet
        mel_outputs_GAN = y_pred[8][0]
        mel_targets, gate_targets, *_ = y
        mel_outputs = mel_outputs[:, :mel_targets.shape[1], :]
        
        plot_n_files = 5
        # plot infer alignment, mel target and predicted, gate predicted
        for idx in range(plot_n_files):# plot longest x audio files
            str_idx = '' if idx == 0 else idx
            self.add_image(
                f"infer_alignment{str_idx}",
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"infer_mel_target{str_idx}",
                plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                f"infer_mel_predicted{str_idx}",
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if mel_outputs_GAN is not None:
                self.add_image(
                    f"mel_predicted_GAN{str_idx}",
                    plot_spectrogram_to_numpy(mel_outputs_GAN[idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
            self.add_image(
                f"infer_gate{str_idx}",
                plot_gate_outputs_to_numpy(
                    gate_targets[idx].data.cpu().numpy(),
                    torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')
    
    def log_teacher_forced_validation(self, reduced_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("teacher_forced_validation.loss", reduced_loss, iteration)
        self.add_scalar("teacher_forced_validation.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("teacher_forced_validation.average_max_attention_weight", avg_prob, iteration)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments, *_ = y_pred
        if mel_outputs_postnet is not None:
            mel_outputs = mel_outputs_postnet
        mel_outputs_GAN = y_pred[8][0]
        mel_targets, gate_targets, *_ = y
        mel_outputs = mel_outputs[:, :mel_targets.shape[1], :]
        mel_MSE_map = torch.nn.MSELoss(reduction='none')(mel_outputs, mel_targets)
        mel_MSE_map[:, -1, -1] = 20.0 # because otherwise the color map scale is crap
        
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        plot_n_files = 5
        # plot alignment, mel target and predicted, gate target and predicted
        for idx in range(plot_n_files):# plot longest x audio files
            str_idx = '' if idx == 0 else idx
            self.add_image(
                f"teacher_forced_alignment{str_idx}",
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"mel_target{str_idx}",
                plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                f"mel_predicted{str_idx}",
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if mel_outputs_GAN is not None:
                self.add_image(
                    f"mel_predicted_GAN{str_idx}",
                    plot_spectrogram_to_numpy(mel_outputs_GAN[idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
            self.add_image(
                f"mel_squared_error{str_idx}",
                plot_spectrogram_to_numpy(mel_MSE_map[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                f"gate{str_idx}",
                plot_gate_outputs_to_numpy(
                    gate_targets[idx].data.cpu().numpy(),
                    torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')
    
    def log_weighted(self, term, iteration, section='', name=''):
        if len(section) and section[-1] != '/':
            section = section + '/'
        self.add_scalar(f"{section}{name}", term[0], iteration)
        if term[1] != 0.0 and term[1] != 1.0:
            self.add_scalar(f"_weighted_{section}{name}", term[0]*term[1], iteration)
    
    def log_additional_losses(self, loss_terms, iteration, prepend=''):
        self.log_weighted(loss_terms[1], iteration, section=f'{prepend}spect', name='MSELoss')
        self.log_weighted(loss_terms[2], iteration, section=f'{prepend}spect', name='L1Loss')
        self.log_weighted(loss_terms[3], iteration, section=f'{prepend}spect', name='SmoothedL1Loss')
        self.log_weighted(loss_terms[4], iteration, section=f'{prepend}postnet', name='MSELoss')
        self.log_weighted(loss_terms[5], iteration, section=f'{prepend}postnet', name='L1Loss')
        self.log_weighted(loss_terms[6], iteration, section=f'{prepend}postnet', name='SmoothedL1Loss')
        self.log_weighted(loss_terms[7], iteration, section='', name=f'{prepend}GateLoss')
        self.log_weighted(loss_terms[8], iteration, section=f'{prepend}SylpsNet', name='KLDivergence')
        self.log_weighted(loss_terms[9], iteration, section=f'{prepend}SylpsNet', name='PredSylpsMSE')
        self.log_weighted(loss_terms[10], iteration, section=f'{prepend}SylpsNet', name='PredSylpsMAE')
        self.log_weighted(loss_terms[11], iteration, section=f'{prepend}SupervisedLoss', name='Total')
        self.log_weighted(loss_terms[12], iteration, section=f'{prepend}SupervisedLoss', name='KLDivergence')
        self.log_weighted(loss_terms[13], iteration, section=f'{prepend}UnsupervisedLoss', name='Total')
        self.log_weighted(loss_terms[14], iteration, section=f'{prepend}UnsupervisedLoss', name='KLDivergence')
        self.log_weighted(loss_terms[15], iteration, section=f'{prepend}ClassicationLoss', name='MSE')
        self.log_weighted(loss_terms[16], iteration, section=f'{prepend}ClassicationLoss', name='MAE')
        self.log_weighted(loss_terms[17], iteration, section=f'{prepend}ClassicationLoss', name='NCE')
        self.log_weighted(loss_terms[18], iteration, section=f'{prepend}AuxClassicationLoss', name='MSE')
        self.log_weighted(loss_terms[19], iteration, section=f'{prepend}AuxClassicationLoss', name='MAE')
        self.log_weighted(loss_terms[20], iteration, section=f'{prepend}AuxClassicationLoss', name='NCE')
        self.log_weighted(loss_terms[21], iteration, section=f'{prepend}PredEmotionNetZuLoss', name='MSE')
        self.log_weighted(loss_terms[22], iteration, section=f'{prepend}PredEmotionNetZuLoss', name='MAE')
        self.log_weighted(loss_terms[24], iteration, section=f'{prepend}AdversarialLoss', name='AvgFakeness')
        self.log_weighted(loss_terms[25], iteration, section=f'{prepend}AdversarialLoss', name='Generator')
        self.log_weighted(loss_terms[26], iteration, section=f'{prepend}AdversarialLoss', name='Discriminator')
        
        self.add_scalar(f'{prepend}ClassicationTop1Acc', loss_terms[23][0], iteration)
        
        #[loss.item(), 1.0],                                                  00
        #[spec_MSE.item(), self.melout_MSE_scalar],                           01
        #[spec_MAE.item(), self.melout_MAE_scalar],                           02
        #[spec_SMAE.item(), self.melout_SMAE_scalar],                         03
        #[postnet_MSE.item(), self.postnet_MSE_scalar],                       04
        #[postnet_MAE.item(), self.postnet_MAE_scalar],                       05
        #[postnet_SMAE.item(), self.postnet_SMAE_scalar],                     06
        #[gate_loss.item(), 1.0],                                             07
        #[sylKLD.item(), self.syl_KDL_weight],                                08
        #[sylps_MSE.item(), self.pred_sylps_MSE_weight],                      09
        #[sylps_MAE.item(), self.pred_sylps_MAE_weight],                      10
        #[SupervisedLoss.item(), 1.0],                                        11
        #[SupervisedKDL.item(), em_kl_weight*0.5],                            12
        #[UnsupervisedLoss.item(), 1.0],                                      13
        #[UnsupervisedKDL.item(), em_kl_weight*0.5],                          14
        #[ClassicationMSELoss.item(), self.zsClassificationMSELoss],          15
        #[ClassicationMAELoss.item(), self.zsClassificationMAELoss],          16
        #[ClassicationNCELoss.item(), self.zsClassificationNCELoss],          17
        #[AuxClassicationMSELoss.item(), self.auxClassificationMSELoss],      18
        #[AuxClassicationMAELoss.item(), self.auxClassificationMAELoss],      19
        #[AuxClassicationNCELoss.item(), self.auxClassificationNCELoss],      20
        #[PredDistMSE.item(), self.predzu_MSE_weight],                        21
        #[PredDistMAE.item(), self.predzu_MAE_weight],                        22
        #[Top1ClassificationAcc, 1.0],                                        23
        #[reduced_avg_fakeness, 1.0],                                         24
        #[adv_postnet_loss.item(), self.adv_postnet_scalar],                  25
        #[reduced_d_loss, self.adv_postnet_scalar],                           26