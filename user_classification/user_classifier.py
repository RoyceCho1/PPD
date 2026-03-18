import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datasets
from absl import app, flags
import wandb
import os
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('epochs', 10, 'Number of epochs for training')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for training')
flags.DEFINE_integer('input_size', 3584, 'Input size of the model') 
flags.DEFINE_integer('hidden_size', 512, 'Hidden size of the model(hidden layer size)')
flags.DEFINE_integer('num_layers', 4, 'Number of layers in the model')
flags.DEFINE_integer('num_classes', 304, 'Number of unique users')
flags.DEFINE_string('dataset_name', 'TODO: add dataset name', 'Name of the model')
flags.DEFINE_string('wandb_project', 'TODO: add wandb project name', 'Wandb project name')
flags.DEFINE_float('warmup_ratio', 0.1, 'Warmup ratio for the scheduler')
flags.DEFINE_float('dropout', 0.1, 'Dropout probability')
flags.DEFINE_integer('num_test_batches', 10, 'Number of test batches to evaluate')
flags.DEFINE_string('output_dir', 'TODO: add output directory', 'Output directory to save the model')

def get_accuracy(probs, class_, k=1):
    if k == 1:
        # Top-1 accuracy (standard classification accuracy)
        pred = torch.argmax(probs, dim=-1) # 제일 값이 높은 index 반환
        correct = torch.sum(pred == class_) # 정답과 일치하는 개수
        return correct.item() / class_.size(0) # 정확도
    else:
        # Top-k accuracy
        top_k = torch.topk(probs, k, dim=-1).indices # 제일 값이 높은 k개의 index 반환 [batch_size, k]
        correct = torch.sum(top_k.eq(class_.unsqueeze(-1))) # 정답과 일치하는 개수(top-k에 해당 index가 있는지 확인)
        return correct.item() / class_.size(0) # 정확도

class UserClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, drop_p=0.1):
        super(UserClassifier, self).__init__()
        
        # Define the model
        self.input_size = input_size # input embedding size
        self.hidden_size = hidden_size # hidden layer size
        self.num_layers = num_layers # number of mlp layers
        
        self.input_layer = nn.Linear(input_size, hidden_size) # input_size 데이터를 받아서 hidden_size로 변환
        self.activation_layer = nn.GELU() # 비선형 함수
        
        self.mlp_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]) # hidden_size 데이터를 받아서 hidden_size로 변환, default 4 layers
        self.mlp_activation_layers = nn.ModuleList([nn.GELU() for _ in range(num_layers)]) # 비선형 함수, default 4 layers
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)]) # layer normalization, default 4 layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(drop_p) for _ in range(num_layers)]) # dropout, default 4 layers
        
        self.output_layer = nn.Linear(hidden_size, num_classes) # hidden_size 데이터를 받아서 num_classes로 변환
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_layer(x)
        
        for i in range(self.num_layers): # default 4 layers
            x = x + self.mlp_layers[i](x)
            x = self.mlp_activation_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.dropout_layers[i](x)
        
        x = self.output_layer(x)    
        
        return x
    
def main(_):
    random_str = np.random.bytes(4).hex()
    unique_run_name = f'{FLAGS.dataset_name}-{FLAGS.batch_size}-{FLAGS.epochs}-{FLAGS.learning_rate}-{FLAGS.weight_decay}-{FLAGS.input_size}-{FLAGS.hidden_size}-{FLAGS.num_layers}'
    unique_run_name = unique_run_name.replace('/', '-').replace(' ', '_').replace('.', '_')
    unique_run_name = f'{unique_run_name}-{random_str}' # unique run name
    
    ds = datasets.load_dataset(FLAGS.dataset_name)
    remove_cols = list(ds['train'].column_names)
    remove_cols.remove('class')
    remove_cols.remove('emb')
    # class하고 emb를 제외한 나머지 column들을 remove_cols에 추가
    def map_fn(examples):
        for i in range(len(examples['emb'])):
            examples['emb'][i] = examples['emb'][i][-1] # last hidden state
        return examples
    ds = ds.map(map_fn, batched=True, num_proc=os.cpu_count(), remove_columns=remove_cols)
    
    train_dataloader = torch.utils.data.DataLoader(
        ds['train'], 
        batch_size=FLAGS.batch_size, 
        shuffle=True,
    ) # batch size로 묶어서 데이터 로드, 데이터는 무작위로 섞어서 로드
    test_dataloader = torch.utils.data.DataLoader(
        ds['test'],
        batch_size=FLAGS.batch_size, 
        shuffle=True,
    ) # batch size로 묶어서 데이터 로드, 데이터는 무작위로 섞어서 로드
    
    model = UserClassifier(
        input_size=FLAGS.input_size,
        hidden_size=FLAGS.hidden_size,
        num_classes=FLAGS.num_classes,
        num_layers=FLAGS.num_layers
    )
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss() # loss function: Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay) # optimizer: Adam
    num_steps = len(train_dataloader) * FLAGS.epochs # number of steps
    warmup_steps = int(FLAGS.warmup_ratio * len(train_dataloader)) # number of warmup steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps) # learning rate scheduler: cosine schedule with warmup
    
    config_dict = FLAGS.flag_values_dict()
    wandb.init(project=FLAGS.wandb_project, config=config_dict)
    
    def process_batch(data):
        emb = torch.stack(data['emb']).T.to(device='cuda', dtype=torch.float) 
        class_ = data['class'].to(device='cuda', dtype=torch.long)
        return emb, class_
    # emb shape: [batch_size, input_size], class_ shape: [batch_size]
    # Tensor로 만들고, cuda로 이동
    
    curr_step = 0
    for epoch in range(FLAGS.epochs):
        for i, data in enumerate(train_dataloader):
            exact_epoch = epoch + i / len(train_dataloader)
            emb, class_ = process_batch(data)
            
            optimizer.zero_grad()
            logits = model(emb) # forward pass
            probs = F.softmax(logits, dim=-1) # softmax
            
            loss = criterion(logits, class_) # loss calculation
            loss.backward() # backward pass
            optimizer.step() # update weights
            scheduler.step() # update learning rate
            
            train_stats = dict()
            
            train_stats['loss'] = loss.item()
            train_stats['accuracy'] = get_accuracy(probs, class_)
            train_stats['top2_accuracy'] = get_accuracy(probs, class_, k=2)
            train_stats['top4_accuracy'] = get_accuracy(probs, class_, k=4)
            train_stats['top8_accuracy'] = get_accuracy(probs, class_, k=8)
            train_stats['top16_accuracy'] = get_accuracy(probs, class_, k=16)
            train_stats['top32_accuracy'] = get_accuracy(probs, class_, k=32)
            train_stats['epoch'] = exact_epoch
            train_stats['lr'] = optimizer.param_groups[0]['lr']
            
            desired_prob = probs[torch.arange(probs.size(0)), class_]
            train_stats['desired_prob'] = desired_prob.mean().item()
            
            train_stats = {f'train/{key}': value for key, value in train_stats.items()}
            wandb.log(train_stats, step=curr_step)
            
            if i % 100 == 0:
                with torch.no_grad():
                    test_stats = defaultdict(float)
                    for j, test_data in enumerate(test_dataloader):
                        emb, class_ = process_batch(test_data)
                        
                        logits = model(emb)
                        probs = F.softmax(logits, dim=-1)
                        
                        loss = criterion(logits, class_)
                        
                        desired_prob = probs[torch.arange(probs.size(0)), class_]
                        accuracy = get_accuracy(probs, class_)
                        
                        test_stats['loss'] += loss.item()
                        test_stats['accuracy'] += accuracy
                        test_stats['top2_accuracy'] += get_accuracy(probs, class_, k=2)
                        test_stats['top4_accuracy'] += get_accuracy(probs, class_, k=4)
                        test_stats['top8_accuracy'] += get_accuracy(probs, class_, k=8)
                        test_stats['top16_accuracy'] += get_accuracy(probs, class_, k=16)
                        test_stats['top32_accuracy'] += get_accuracy(probs, class_, k=32)
                        test_stats['desired_prob'] += desired_prob.mean().item()
                        test_stats['num_batches'] += 1
                        
                        if test_stats['num_batches'] == FLAGS.num_test_batches:
                            break
                    
                    for key in test_stats:
                        if key != 'num_batches':
                            test_stats[key] /= test_stats['num_batches']
                    log_dict = {f'test/{key}': value for key, value in test_stats.items()}  
                    print(f'Epoch: {exact_epoch}, Step: {i}, Loss: {loss.item()}, Accuracy: {accuracy}, Test Loss: {test_stats["loss"]}, Test Accuracy: {test_stats["accuracy"]}')
                    wandb.log(log_dict, step=curr_step)
            else:
                print(f'Epoch: {exact_epoch}, Step: {i}, Loss: {loss.item()}, Accuracy: {accuracy}')
            curr_step += 1
        
        # Save model
        output_dir = os.path.join(FLAGS.output_dir, unique_run_name, f'epoch_{epoch}')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pth'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pth'))
        
    wandb.finish()
    
    

if __name__ == '__main__':
    app.run(main)