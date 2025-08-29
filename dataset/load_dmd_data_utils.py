import pickle


def load_steps_data_from_file(file_path: str):
    """
    从单个文件加载steps数据
    
    Args:
        file_path: steps数据文件路径
    
    Returns:
        steps数据列表
    """
    with open(file_path, 'rb') as f:
        steps_data = pickle.load(f)
    return steps_data


file_path = "/mnt/hz_trainer/batch_output_20250827_112221/steps_data/steps_20250827_112221_prompt000_sample01_seed577814209.pkl"
data = load_steps_data_from_file(file_path)
sigmas,latent = [data[i]["data"]["sigma"].item() for i in range(len(data))]
# sort sigmas
sigmas.sort()
print(sigmas)
#



