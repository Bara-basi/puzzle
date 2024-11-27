import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def calculate_consistency_loss(predictions):
    # 使用softmax获取位置概率分布
    probs = torch.softmax(predictions, dim=-1)
    
    # 计算每个位置被多次分配的惩罚
    duplicates = torch.sum(probs, dim=1) - 1.0
    return torch.mean(torch.abs(duplicates))

def calculate_accuracy(predictions, targets):
    
    # 获取每个位置预测的最大概率的索引
    pred_positions = torch.argmax(predictions, dim=-1)  # [B, 9]
    
    # 计算单块准确率 (每个位置预测正确的比例)
    block_correct = (pred_positions == targets).float()
    block_accuracy = block_correct.mean().item()
    
    # 计算完整拼图准确率 (所有位置都正确的比例)
    puzzle_correct = (block_correct.sum(dim=1) == 9).float()
    puzzle_accuracy = puzzle_correct.mean().item()
    
    # 计算每个位置的准确率
    position_accuracy = block_correct.mean(dim=0).cpu().numpy()
    
    return block_accuracy, puzzle_accuracy, position_accuracy

def evaluate(model, dataloader, device, epoch=-1):
    model.eval()
    total_position_loss = 0
    total_consistency_loss = 0
    total_samples = 0
    
    # 用于累积准确率
    total_block_acc = 0
    total_puzzle_acc = 0
    position_accs = np.zeros(9)
    
    show_one_output = True
    # 使用tqdm显示进度条
    with torch.no_grad():
        for blocks, positions in tqdm(dataloader, desc=f'Evaluating epoch {epoch}'):
            blocks = blocks.to(device)
            positions = positions.to(device)
            predictions = model(blocks)
            if show_one_output:
                print(f"y_pred: {predictions[-1].argmax(dim=-1).cpu().numpy()}")
                print(f"y_true: {positions[-1].cpu().numpy()}")
                show_one_output = False
            # 计算位置分类损失
            position_loss = nn.functional.cross_entropy(predictions.view(-1, 9), positions.view(-1))
            total_position_loss += position_loss.item() * blocks.size(0)
            
            # 计算一致性损失
            consistency_loss = calculate_consistency_loss(predictions)
            total_consistency_loss += consistency_loss.item() * blocks.size(0)
            
            # 计算准确率
            block_acc, puzzle_acc, pos_acc = calculate_accuracy(predictions, positions)
            total_block_acc += block_acc * blocks.size(0)
            total_puzzle_acc += puzzle_acc * blocks.size(0)
            position_accs += pos_acc * blocks.size(0)
            total_samples += blocks.size(0)
    
    # 计算平均值
    metrics = {
        'position_loss': total_position_loss / total_samples,
        'consistency_loss': total_consistency_loss / total_samples,
        'block_accuracy': total_block_acc / total_samples,
        'puzzle_accuracy': total_puzzle_acc / total_samples,
        'position_accuracies': position_accs / total_samples
    }
    
    # 打印详细信息
    print("\n" + "="*50)
    if epoch >= 0:
        print(f"Epoch {epoch} Evaluation Results:")
    print(f"Position Loss: {metrics['position_loss']:.4f}")
    print(f"Consistency Loss: {metrics['consistency_loss']:.4f}")
    print(f"Block Accuracy: {metrics['block_accuracy']:.4f}")
    print(f"Complete Puzzle Accuracy: {metrics['puzzle_accuracy']:.4f}")
    print("\nPosition-wise Accuracies:")
    for i, acc in enumerate(metrics['position_accuracies']):
        print(f"Position {i}: {acc:.4f}")
    print("="*50 + "\n")
    
    return total_position_loss / total_samples, total_consistency_loss / total_samples
