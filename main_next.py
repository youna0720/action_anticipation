import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import pdb
import wandb
import random

from einops import repeat, rearrange

from opts import parser
from utils import *
#from datasets.breakfast import *
#from datasets.batch_gen import *
from datasets.EKdataset import *
#from metric import *

#from models.next_transformer import Transformer_next
from models.DETR_next import Detr_next
#from rulstm_utils import topk_recall

#from models.future_prediction import AVTh

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from predict import *
from os.path import join
from torch.utils.data import DataLoader
from torch.backends import cudnn



seed = 38270
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark, cudnn.deterministic = False, True
#os.environ['CUDA_VISIBLE_DEVICES']="4, 5, 6, 7"

#print("current cpu random seed", torch.initial_seed())
#print("current cuda random seed", torch.cuda.initial_seed())
#print("np random seed", np.random.seed())

device = torch.device('cuda')

def main():
    global args
    args = parser.parse_args()

    #Print Running Device
    #if args.cpu:
    #    device = torch.device('cpu')
    #    print('using cpu')
    #else:
    #    device = torch.device('cuda')
    #    print('using gpu')
    device = torch.device('cuda')

    #Print basic information
    print('runs : ', args.runs)
    print('model type : ', args.model)
    print('input type : ', args.input_type)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    #WANDB
    wandb.init(project="next_anticipation", reinit=True)
    wandb.run.name = args.wandb
    wandb.config.update(vars(args))

    #split
    split = args.split

    #Dataset related information
    if args.dataset == 'ek55':
        n_class = 2513
        pad_idx = 2515
        input_seq_len = args.input_seq_len
        dim_in_features = 1024
    if args.dataset == 'ek100':
        n_class = 3806
        pad_idx = 3808
        input_seq_len = args.input_seq_len
        dim_in_features = 1024


    #Declare model
    embed_size = args.hidden_dim
    #embed_size = 24                 #??????
    if args.model == 'detr':
        print('DETR running')
#        decoder_token = args.Ta / args.time_step        #ex) Ta=1, then 4 token when time_step=0.25
        input_seq_len = args.input_seq_len
        model = Detr_next(n_class, embed_size, pad_idx, device=device, args=args, nheads=args.nhead, \
                          num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer, \
                          dropout=args.dropout, input_seq_len=input_seq_len, feature_dim=1024).to(device)

    elif args.model == 'avth':
        print('AVTh running')
        gpt_args = dict()
        gpt_args['n_head'] = args.nhead
        gpt_args['n_layer'] = args.n_avth_layer
        gpt_args['future_pred_loss_wt'] = 1.0
        hidden_dim = args.hidden_dim

        model = AVTh(in_features=dim_in_features, output_len=1, output_len_eval=-1, avg_last_n=1, inter_dim=hidden_dim, \
                     return_past_too=True, n_class=n_class, **gpt_args)
#        model = AVTh(in_features=1024, output_len=1, output_len_eval=-1, avg_last_n=1, inter_dim=512, return_past_too=True, **gpt_args)
#        model = AVTh(in_features=1024, output_len=1, output_len_eval=-1, avg_last_n=1, inter_dim=args.hidden_dim, drop_rate=args.dropout, return_past_too=True, **gpt_args)
    else:
        print('Next Tran running')
        model = Transformer_next(n_class, embed_size, pad_idx, device=device, args=args, nheads=args.nhead, num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer, dropout=args.dropout).to(device)

    #PATH for Saving Results (Model, Result)
    model_save_path = os.path.join('./save_dir/'+args.dataset+'/models/runs'+str(args.runs))
    results_save_path = os.path.join('./save_dir/'+args.dataset+'/results/runs'+str(args.runs))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print('model_save_path : ', model_save_path)
    model_save_file = os.path.join(model_save_path, 'checkpoint.ckpt')

    #Data Parallel
    model = nn.DataParallel(model).to(device)
    wandb.watch(model)

    #Set Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    #CosineAnnealing Learning rate
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')

    #Dataset Loader
    #Using feature loss needs features in Ta
    #For example, since we use 4 tokens for decoder, 3 GT features are needed
    if args.feature_loss or args.feature_one_loss:
        train_frame = args.frame + (args.Ta / args.time_step)
        train_Ta = 0
    else:
        train_frame = args.frame
        train_Ta = args.Ta

    if args.dataset == 'ek55' or args.dataset == 'ek100':
        csv_name = 'training.csv'
        label_type = 'action'

        kargs = {
            'path_to_lmdb': args.path_to_lmdb,
            'path_to_csv': join(args.path_to_data, csv_name),
            'num_class': n_class,
            'time_step': args.time_step,
            'img_tmpl': args.img_tmpl,
            'action_samples': None,
            'past_features': True,
            'sequence_length': train_frame,
            'label_type': label_type,
            'challenge': args.predict,
            'ta': train_Ta,
            'pad_idx': pad_idx
        }
        trainset = SequenceDataset(**kargs)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
#        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, collate_fn=trainset.my_collate)
        csv_name = 'validation.csv'
        label_type = 'action'
        kargs = {
            'path_to_lmdb': args.path_to_lmdb,
            'path_to_csv': join(args.path_to_data, csv_name),
            'num_class': n_class,
            'time_step': args.time_step,
            'img_tmpl': args.img_tmpl,
            'action_samples': None,
            'past_features': True,
            'sequence_length': args.frame,
            'label_type': label_type,
            'challenge': args.predict,
            'ta': args.Ta,
            'pad_idx': pad_idx
        }
        valset = SequenceDataset(**kargs)
        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
#        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False, collate_fn=valset.my_collate)

    #Run the model
    if args.eval:
        model_path = './save_dir/'+args.dataset+'/models/runs'+str(args.runs)+'/checkpoint.ckpt'
        print("Evaluation with model:"+model_path)
        model.load_state_dict(torch.load(model_path))
        evaluate(n_class, model, val_loader, pad_idx)

    elif args.predict:
        print("Prediction with model")
        print("Not implemented yet")
    else:
        model = train(n_class, model, train_loader, val_loader, optimizer, scheduler, criterion, args.epochs, 'transformer', model_save_path, pad_idx)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(), model_save_file)

def train(n_class, model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, model_type, model_save_path, pad_idx):
    model.to(device)
    model.train()
    print("Training Start")
    max_action_seq = args.max_action_seq

    #Iterate epoch
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_loss_next = 0
        epoch_loss_next_dec = 0
        epoch_loss_seg = 0
        epoch_loss_feat = 0
        total_next = 0
        total_next_dec = 0
        total_next_correct = 0
        total_next_dec_correct = 0
        total_seg = 0
        total_seg_correct = 0
        total_seg_acc = 0
#        evaluate(n_class, model, val_loader, pad_idx)

        #FOR each data in train_loader
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label_, target = data['features'], data['past_label'], data['target']
            #features = [32,8,1024]   past_label = [32 8]
            features = features.to(device) #[B, S, C]
            past_label = past_label_.to(device) #[B, S]
            #pdb.set_trace()
            target = target.to(device) # [B]
            target_ = target.unsqueeze(1) # [B,1]
            past_label = F.one_hot(target_.long(), num_classes = n_class).permute(0, 2, 1)
            
            #안하는게맞는듯?.permute(0, 2, 1)
            #cut last few frames for feature loss
            if args.feature_loss:
                cut = (args.Ta/args.time_step)      # 4 when Ta=1, 0.25
                cut *= -1
                cut = int(cut)
                gt_feature = features[:,cut:,:]     #[B, 4, C]
                features = features[:,:cut,:]       #[B, S, C]
                past_emb = past_emb[:,:cut]         #[B, S]
                past_label = past_label[:,:cut]     #[B, S]

            elif args.feature_one_loss:
                cut = (args.Ta/args.time_step)
                cut *= -1
                cut = int(cut)
                gt_feature = features[:,-1:,:]
                features = features[:,:cut,:]
                past_emb = past_emb[:,:cut]
                past_label = past_label[:,:cut]
            
            if args.model == 'avth':
                outputs = model(features)
            else:
                inputs = features
                outputs = model(past_label, inputs)
            losses = 0
            alpha = 1.0
            beta = 1.0
            gamma = 1.0

            if args.feat_loss:
                output_feat = outputs['feat']
                loss_feat = F.mse_loss(output_feat[:,:-1,:], features[:,1:,:])
                losses += alpha*loss_feat
                epoch_loss_feat += loss_feat.item()

            if args.model == 'avth':
                feat_loss = torch.mean(outputs['feat_loss'])
                losses += feat_loss
                epoch_loss_feat += feat_loss.item()

            if args.next :
                output_next = outputs['next']
                if args.model != 'avth':
                    output_next = output_next.squeeze(1) #[B, n_actions]
#                target = target.squeeze()
                loss_next, n_next_correct, n_next_total = cal_performance(output_next, target, pad_idx)
                losses += beta*loss_next
                total_next += n_next_total
                total_next_correct += n_next_correct
                epoch_loss_next += loss_next.item()
            if args.next_dec :
                output_next_dec = outputs['next_dec']
                output_next_dec = output_next_dec.squeeze(1)
                loss_next_dec, n_next_dec_correct, n_next_dec_total = cal_performance(output_next_dec, target[:, 0], pad_idx)
                losses += loss_next_dec
                total_next_dec += n_next_dec_total
                total_next_dec_correct += n_next_dec_correct
                epoch_loss_next_dec += loss_next_dec.item()
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                target_seg = past_label.view(-1).to(device)
#                target_seg = past_label.view(-1)
#                seg_target = past_label.contiguous().reshape(-1).to(device)
#                loss_seg, n_seg_correct, n_seg_total = cal_seg_performance(output_seg, seg_target, pad_idx)

                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, target_seg, -100)
                losses += gamma*loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()


            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.model == 'avth' or args.feat_loss:
            epoch_loss_feat = epoch_loss_feat / (i+1)
            wandb.log({"feat loss":epoch_loss_feat})
            print('epoch loss :%.3f'%epoch_loss_feat)
        if args.next :
            acc_next = total_next_correct / total_next
            epoch_loss_next = epoch_loss_next / (i+1)
            wandb.log({"next acc":acc_next, "next loss":epoch_loss_next})
            print('next loss :%.3f'%epoch_loss_next, ', next acc : %.5f'%acc_next)
        if args.next_dec :
            acc_next_dec = total_next_dec_correct / total_next_dec
            epoch_loss_next_dec = epoch_loss_next_dec / (i+1)
            wandb.log({"next acc":acc_next_dec, "next loss":epoch_loss_next_dec})
            print('next loss :%.3f'%epoch_loss_next_dec, ', next acc : %.5f'%acc_next_dec)
        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            wandb.log({"seg acc":acc_seg, "seg loss": epoch_loss_seg})
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)
        if args.feature_loss:
            wandb.log({"feature loss":epoch_loss_feat})
            print('feature loss:%.3f'%epoch_loss_feat)
        wandb.log({"total loss":epoch_loss})

       #evaluation
        if epoch % 1 == 0 :
            evaluate(n_class, model, val_loader, pad_idx)

        scheduler.step()

        save_path = os.path.join(model_save_path)
        save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')

    return model

def evaluate(n_class, model, test_loader, pad_idx):
    print("Evaluate with the model : ", args.model)
    model.to(device)
    model.eval()
    max_action_seq = args.max_action_seq
    with torch.no_grad():
        total_loss = 0
        total_loss_next = 0
        total_loss_seg = 0
        total_next = 0
        total_next_correct = 0
        total_top5_correct = 0
        total_seg = 0
        total_seg_correct = 0
        for i, data in enumerate(test_loader):
            features, past_label_, target = data['features'], data['past_label'], data['target']
            features = features.to(device) #[B, S, C]
            target = target.to(device)
#            past_emb = past_emb.to(device) #[B, S, 48]
#            future_label = future_label.to(device) #[B, T]
            past_label = past_label_.to(device)
            #past_label = F.one_hot(past_label_.long(), num_classes = n_class).permute(0, 2, 1)  #.permute(0, 2, 1) 할필요없이 B T C 되는듯
            target = target.to(device) # [B]
            '''
        (Pdb) target        [32]
        tensor([ 860, 1976,  762,  881,   85,  359, 1977, 1472,  902,  901, 2093, 1938,
        1792,  213,  762, 1784,  910,  194,  888, 1305, 2371, 1336, 1792, 1494,
        1499, 1726, 1189, 1447, 1761,  855,  723, 2152], device='cuda:0')
        (Pdb) output_next     [32 2513]
        tensor([[-0.2301, -1.8543,  1.0865,  ...,  0.9317,  2.7422,  1.4272],
        [ 0.6934, -0.6131, -0.2714,  ..., -0.7254,  1.0088,  1.4487],
        [-0.1148, -2.5516,  1.7168,  ...,  0.9703, -0.1465,  2.7365],
        ...,
        [ 0.8040, -1.0138, -1.2259,  ...,  1.4637,  0.7821, -1.1180],
        [-0.6233, -0.3507,  1.9519,  ..., -0.9249,  3.7468,  2.4355],
        [ 1.1732, -0.8626,  0.1841,  ..., -0.2222, -0.9874, -2.1301]],
       
       근데 다 이렇게 나옴 [1.7893e-04, 1.8933e-04, 3.8907e-04,  ..., 1.1510e-04, 3.8040e-04,9.7278e-04]
            '''
            target_ = target.unsqueeze(1) # [B,1]
            past_label = F.one_hot(target_.long(), num_classes = n_class).permute(0, 2, 1)
            
#            if args.dataset == 'ek55':
#                future_label = future_label.unsqueeze(1)
#                trans_future = trans_future.unsqueeze(1)

#            inputs = (features, past_emb)
            
            inputs = features
            output = model.module.ddim_sample(past_label, inputs, i)
            #if mode == 'decoder-agg':
            #output = [model.ddim_func(features.to(device), seed) 
            #            for i in range(len(features))] # output is a list of tuples
            #output = [i.cpu() for i in output]
                #left_offset = self.sample_rate // 2
                #right_offset = (self.sample_rate - 1) // 2   
                
            #if mode == 'decoder-noagg':  # temporal aug must be true
            #    output = [model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
            #    output = [i.cpu() for i in output]
                #left_offset = self.sample_rate // 2
                #right_offset = 0
    
            '''
            inputs = features
            outputs = model(past_label, inputs)   #  32 1 512
            '''
            

#            if mode == 'oneshot' :
#                outputs = model(inputs)
#            elif mode == 'ar':
#                outputs = model(inputs=inputs, mode='eval')
            losses = 0
            if args.next :
                output_next = output         #[B 1 512]                     ddim_sample 출력 [32 2513 1]
                #output_next = outputs['next']
                #output_next = output_next.squeeze(1) #[B, n_actions] 
                #pdb.set_trace()
                output_next = rearrange(output_next, 'b c t -> b t c')  
                output_next = output_next.squeeze(dim=1) #[B, n_actions]       [32 2513]    이거다시체크
                '''
                [[1.7893e-04, 1.8933e-04, 3.8907e-04,  ..., 1.1510e-04, 3.8040e-04,9.7278e-04],
        [1.4791e-04, 1.9458e-04, 5.7390e-04,  ..., 1.0049e-04, 4.6688e-04, 1.1466e-03],
        [2.4500e-04, 1.6630e-04, 5.2005e-04,  ..., 9.7066e-05, 4.0033e-04,  1.1756e-03],
        ...,
        [3.2574e-04, 1.9756e-04, 2.9036e-04,  ..., 1.2529e-04, 5.1749e-04, 9.6893e-04],
        [2.0841e-04, 1.9971e-04, 4.4113e-04,  ..., 1.0929e-04, 4.3300e-04, 1.3157e-03],
        [2.1932e-04, 1.9696e-04, 3.8263e-04,  ..., 1.3292e-04, 5.1913e-04, 1.3593e-03]]
                '''
                
                
                #if i==0:
                #    print(output_next.max(1))
                #recall = topk_recall(Tensor.cpu(output_next), Tensor.cpu(target), k=5)
#                action_recalls = topk_recall_multiple_timesteps(output_next.detach().cpu(), target.detach().cpu())

                n_top5_correct = cal_top5(output_next, target, pad_idx)
#                top5_recall = topk_recall(output_next, target, 5)
                loss_next, n_next_correct, n_next_total = cal_performance(output_next, target, pad_idx)
                losses += loss_next
                total_next += n_next_total
                total_next_correct += n_next_correct
                total_top5_correct += n_top5_correct
                total_loss_next += loss_next.item()
            if args.seg or args.feat_seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C)
                past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, past_label, -100)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                total_loss_seg += loss_seg.item()


#        total_loss = total_loss / (i+1)
#        first_acc = first_correct / (n_first)
#        remain_acc = (total_class_correct - first_correct) / (total_class - n_first)
#        print("sos acc : %.3f" %first_acc, first_correct, n_first)
#        print("remain acc : %.3f" %remain_acc, total_class_correct - first_correct , total_class-n_first)
        if args.next :
            acc_next = total_next_correct / total_next
            acc_top5 = total_top5_correct / total_next
            total_loss_next = total_loss_next / (i+1)
            wandb.log({"val next acc":acc_next, "val top5 next acc":acc_top5, "val next loss":total_loss_next})
            print('next loss :%.3f'%total_loss_next, ', next acc : %.5f'%acc_next, ', top5 acc : %.5f'%acc_top5)
        if args.seg :
            acc_seg = total_seg_correct / total_seg
            total_loss_seg = total_loss_seg / (i+1)
            wandb.log({"val seg acc":acc_seg, "val seg loss": total_loss_seg})
            print('val seg loss :%.3f'%total_loss_seg, ', val seg acc : %.5f'%acc_seg)

if __name__ == '__main__':
    main()
