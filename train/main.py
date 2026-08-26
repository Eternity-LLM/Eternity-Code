import os
import torch
import torch.distributed as dist
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import Qwen3ForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from modeling.modeling import MTP
from trainer import *

def setup_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.npu.set_device(local_rank)

    dist.init_process_group(backend='hccl', init_method='env://')
    
    return local_rank

def load_model_and_tokenizer(local_rank, qwen3path='Qwen/Qwen3-8B'):
    tokenizer = AutoTokenizer.from_pretrained(qwen3path, trust_remote_code=True)
    
    model = Qwen3ForCausalLM.from_pretrained(
        qwen3path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).npu()

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    

    model.module.gradient_checkpointing_enable()
    
    return model, tokenizer

'''def create_dataloader(tokenizer, local_rank, world_size):
    """创建分布式数据加载器"""
    # 这里是示例数据，实际替换成你的数据集
    dummy_texts = ["Hello world"] * 1000
    encodings = tokenizer(dummy_texts, return_tensors='pt', padding=True, truncation=True)
    
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # 每卡batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, sampler

'''

def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()

    is_main = (local_rank == 0)
    
    if is_main:
        print(f"World size: {world_size}")
        print(f"Backend: {dist.get_backend()}")
        print(f"NPU available: {torch.npu.is_available()}")
    

    model, tokenizer = load_model_and_tokenizer(local_rank)
    
'''
    dataloader, sampler = create_dataloader(tokenizer, local_rank, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 5. 训练循环
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)  # 重要：每个epoch重置采样器
        
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(f'npu:{local_rank}')
            attention_mask = attention_mask.to(f'npu:{local_rank}')
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 只在主进程打印日志
            if is_main and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 6. 保存checkpoint（用module提取原始模型）
    if is_main:
        torch.save(model.module.state_dict(), "qwen3_finetuned.pth")
        print("Model saved!")
    
    # 7. 清理
    dist.destroy_process_group()'''

if __name__ == "__main__":
    main()