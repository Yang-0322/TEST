import torch
import time
import sys

def occupy_gpu(target_gpu, target_gb):
    """固定占用指定显卡的显存"""
    target_mb = target_gb * 1024
    chunk_mb = 256
    elements_per_chunk = (chunk_mb * 1024 * 1024) // 4
    
    tensors = []
    print(f"🚀 任务结束，开始抢占 GPU:{target_gpu}，目标: {target_gb}GB")
    
    try:
        while (len(tensors) * chunk_mb) < target_mb:
            t = torch.empty(elements_per_chunk, dtype=torch.float32, device=f'cuda:{target_gpu}')
            t.fill_(1.0) # 物理写入
            tensors.append(t)
            if len(tensors) % 4 == 0:
                print(f"已占用: {len(tensors) * chunk_mb} MB")
        
        print(f"✅ 成功锁定 {target_gb}GB。按 Ctrl+C 释放。")
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("\n释放显存...")
    except Exception as e:
        print(f"抢占失败: {e}")

    print("train程序停止运行")

if __name__ == "__main__":
    # 可以通过命令行参数传入：python hold_gpu.py [GPU_ID] [GB]
    gpu_id = 0
    gb_to_hold = 15
    occupy_gpu(gpu_id, gb_to_hold)
