import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

class MRCTDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        """
        root: 資料根目錄，包含 'Train', 'Val', 'Test' 資料夾
        mode: 'Train', 'Val', 'Test' 來選擇數據集
        transform: 可選的數據增強操作
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # 設定 MRI 和 CT 對應的資料夾路徑
        self.mr_path = os.path.join(self.root, self.mode, 'MR')
        self.ct_path = os.path.join(self.root, self.mode, 'CT')
        
        # 獲取 MRI 和 CT 的 .nii 檔案
        self.mr_files = sorted([os.path.join(self.mr_path, f) for f in os.listdir(self.mr_path) if f.endswith('.nii')])
        self.ct_files = sorted([os.path.join(self.ct_path, f) for f in os.listdir(self.ct_path) if f.endswith('.nii')])
        
        # 檢查 MRI 和 CT 文件數量是否一致
        assert len(self.mr_files) == len(self.ct_files), "MRI 和 CT 文件數量不匹配"
        
    def __len__(self):
        return len(self.mr_files)

    def __getitem__(self, idx):
        # 讀取 MRI 和 CT 影像
        mr_image = sitk.ReadImage(self.mr_files[idx])
        ct_image = sitk.ReadImage(self.ct_files[idx])

        # 將 MRI 和 CT 影像轉為 numpy array 並且設定數據類型
        mr_array = np.array(sitk.GetArrayFromImage(mr_image), dtype=np.float32)  # MRI 數據 (192, 192, 7)
        ct_array = np.array(sitk.GetArrayFromImage(ct_image), dtype=np.float32)  # CT 數據 (192, 192)

        # 調整 MRI 數據的形狀，將 (192, 192, 7) 轉換為 (7, 192, 192)
        mr_array = np.transpose(mr_array, (2, 0, 1))  # 調整為 (7, 192, 192)

        # 確保 CT 數據是 2D 並有 1 個通道 (batch_size, 1, 192, 192)
        if len(ct_array.shape) == 3:  # 如果 CT 影像的形狀是 (192, 192, 1)
            ct_array = ct_array.squeeze(-1)  # 去掉多餘的單一維度

        # 如果需要數據增強或其他預處理
        if self.transform:
            mr_array = self.transform(mr_array)
            ct_array = self.transform(ct_array)

        # 將 MRI 和 CT 數據轉為 PyTorch 張量
        mr_tensor = torch.from_numpy(mr_array).float()  # MRI (7, 192, 192)
        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0)  # CT (1, 192, 192)
        
        return mr_tensor, ct_tensor



def get_dataloaders(root_dir, batch_size=4):
    """
    root_dir: 資料根目錄
    batch_size: DataLoader 的批次大小
    """
    train_dataset = MRCTDataset(root=root_dir, mode='Train')
    val_dataset = MRCTDataset(root=root_dir, mode='Val')
    test_dataset = MRCTDataset(root=root_dir, mode='Test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 設置資料路徑，確保該路徑下有 Train, Val, Test 資料夾
    root_dir = "Data_2D"
    
    # 創建 DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(root_dir)

    # 檢查訓練集大小
    print(f"訓練集大小: {len(train_loader.dataset)}")
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"測試集大小: {len(test_loader.dataset)}")

    # 測試是否可以正確加載一個批次的數據
    for mr, ct in train_loader:
        print(f"MRI 影像大小: {mr.shape}")  # 預期大小: (batch_size, 7, 192, 192)
        print(f"CT 影像大小: {ct.shape}")   # 預期大小: (batch_size, 1, 192, 192)
        break  # 只加載一個批次進行檢查
