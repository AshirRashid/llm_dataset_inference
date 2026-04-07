gptq_model_path = "/scratch/ar7789/llm_quant/quant/saved_qmodels/EleutherAI/pythia-160m-deduped-gptq/"
model_name = "EleutherAI/pythia-160m-deduped"

cache_dir = "/scratch/ar7789/.cache/huggingface"
# list_number_samples = [2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
list_number_samples = [1000]

from utils import prepare_model
from metrics import aggregate_metrics
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2, norm
import torch
import torch.nn as nn
from tqdm import tqdm
from selected_features import feature_list


def split_train_val(metrics):
    keys = list(metrics.keys())
    num_elements = len(metrics[keys[0]])
    print (f"Using {num_elements} elements")
    # select a random subset of val_metrics (50% of ids)
    ids_train = np.random.choice(num_elements, num_elements//2, replace=False)
    ids_val = np.array([i for i in range(num_elements) if i not in ids_train])
    new_metrics_train = {}
    new_metrics_val = {}
    for key in keys:
        new_metrics_train[key] = np.array(metrics[key])[ids_train]
        new_metrics_val[key] = np.array(metrics[key])[ids_val]
    return new_metrics_train, new_metrics_val

def remove_outliers(metrics, remove_frac=0.05, outliers = "zero"):
    # Sort the array to work with ordered data
    sorted_ids = np.argsort(metrics)
    
    # Calculate the number of elements to remove from each side
    total_elements = len(metrics)
    elements_to_remove_each_side = int(total_elements * remove_frac / 2) 
    
    # Ensure we're not attempting to remove more elements than are present
    if elements_to_remove_each_side * 2 > total_elements:
        raise ValueError("remove_frac is too large, resulting in no elements left.")
    
    # Change the removed metrics to 0.
    lowest_ids = sorted_ids[:elements_to_remove_each_side]
    highest_ids = sorted_ids[-elements_to_remove_each_side:]
    all_ids = np.concatenate((lowest_ids, highest_ids))

    # import pdb; pdb.set_trace()
    
    trimmed_metrics = np.copy(metrics)
    
    if outliers == "zero":
        trimmed_metrics[all_ids] = 0
    elif outliers == "mean" or outliers == "mean+p-value":
        trimmed_metrics[all_ids] = np.mean(trimmed_metrics)
    elif outliers == "clip":
        highest_val_permissible = trimmed_metrics[highest_ids[0]]
        lowest_val_permissible = trimmed_metrics[lowest_ids[-1]]
        trimmed_metrics[highest_ids] =  highest_val_permissible
        trimmed_metrics[lowest_ids] =   lowest_val_permissible
    elif outliers == "randomize":
        #this will randomize the order of metrics
        trimmed_metrics = np.delete(trimmed_metrics, all_ids)
    else:
        assert outliers in ["keep", "p-value"]
        pass
        
    return trimmed_metrics

def normalize_and_stack(train_metrics, val_metrics, normalize="train"):
    '''
    excpects an input list of list of metrics
    normalize val with corre
    '''
    new_train_metrics = []
    new_val_metrics = []
    for (tm, vm) in zip(train_metrics, val_metrics):
        if normalize == "combined":
            combined_m = np.concatenate((tm, vm))
            mean_tm = np.mean(combined_m)
            std_tm = np.std(combined_m)
        else:
            mean_tm = np.mean(tm)
            std_tm = np.std(tm)
        
        if normalize == "no":
            normalized_vm = vm
            normalized_tm = tm
        else:
            #normalization should be done with respect to the train set statistics
            normalized_vm = (vm - mean_tm) / std_tm
            normalized_tm = (tm - mean_tm) / std_tm
        
        new_train_metrics.append(normalized_tm)
        new_val_metrics.append(normalized_vm)

    train_metrics = np.stack(new_train_metrics, axis=1)
    val_metrics = np.stack(new_val_metrics, axis=1)
    return train_metrics, val_metrics

def prepare_metrics(members_metrics, nonmembers_metrics, outliers="clip", return_tensors=False):
    keys = list(members_metrics.keys())
    np_members_metrics = []
    np_nonmembers_metrics = []
    for key in keys:
        members_metric_key = np.array(members_metrics[key])
        nonmembers_metric_key = np.array(nonmembers_metrics[key])
        
        if outliers is not None:
            # remove the top 2.5% and bottom 2.5% of the data
            members_metric_key = remove_outliers(members_metric_key, remove_frac = 0.05, outliers = outliers)
            nonmembers_metric_key = remove_outliers(nonmembers_metric_key, remove_frac = 0.05, outliers = outliers)

        np_members_metrics.append(members_metric_key)
        np_nonmembers_metrics.append(nonmembers_metric_key)

    # concatenate the train and val metrics by stacking them
    np_members_metrics, np_nonmembers_metrics = normalize_and_stack(np_members_metrics, np_nonmembers_metrics)
    if return_tensors:
        np_members_metrics = torch.tensor(np_members_metrics, dtype=torch.float32)
        np_nonmembers_metrics = torch.tensor(np_nonmembers_metrics, dtype=torch.float32)

    return np_members_metrics, np_nonmembers_metrics

# aux functions about MIA classifier
def get_dataset_splits(_train_metrics, _val_metrics, num_samples):
    # get the train and val sets
    for_train_train_metrics = _train_metrics[:num_samples]
    for_train_val_metrics = _val_metrics[:num_samples]
    for_val_train_metrics = _train_metrics[num_samples:]
    for_val_val_metrics = _val_metrics[num_samples:]


    # create the train and val sets
    train_x = np.concatenate((for_train_train_metrics, for_train_val_metrics), axis=0)
    train_y = np.concatenate((-1*np.zeros(for_train_train_metrics.shape[0]), np.ones(for_train_val_metrics.shape[0])))
    val_x = np.concatenate((for_val_train_metrics, for_val_val_metrics), axis=0)
    val_y = np.concatenate((-1*np.zeros(for_val_train_metrics.shape[0]), np.ones(for_val_val_metrics.shape[0])))
    
    # return tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    
    return (train_x, train_y), (val_x, val_y)

def train_model(inputs, y, num_epochs=10000):
    num_features = inputs.shape[1]
    model = get_model(num_features)
        
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert y to float tensor for BCEWithLogitsLoss
    y_float = y.float()

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Squeeze the output to remove singleton dimension
            loss = criterion(outputs, y_float)
            loss.backward()
            optimizer.step()
            pbar.set_description('loss {}'.format(loss.item()))
    return model

def get_model(num_features, linear = True):
    if linear:
        model = nn.Linear(num_features, 1)
    else:
        model = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Single output neuron
        )
    return model

def get_predictions(model, val, y):
    with torch.no_grad():
        preds = model(val).detach().squeeze()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(preds, y.float())
    return preds.numpy(), loss.item()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, val, y):
    # get auc and plot roc curve
    from sklearn.metrics import roc_auc_score
    preds, _ = get_predictions(model, val, y)
    auc_score = roc_auc_score(y, preds)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y, preds)
    
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return auc_score

def get_p_value_list(heldout_train, heldout_val, list_number_samples):
    # list_number_samples is used to see how the p-values changes across different number of samples
    if (not np.isfinite(heldout_train).all()) or (not np.isfinite(heldout_val).all()):
        print("WARNING - CUSTOM: There is an infinite value in one of the populations")

    # filter for infinite values to avoid a nan p_value
    heldout_train = heldout_train[np.isfinite(heldout_train)]
    heldout_val = heldout_val[np.isfinite(heldout_val)]
    
    p_value_list = []
    for num_samples in list_number_samples:
        heldout_train_curr = heldout_train[:num_samples]
        heldout_val_curr = heldout_val[:num_samples]
        t, p_value = ttest_ind(heldout_train_curr, heldout_val_curr, alternative='less')
        p_value_list.append(p_value)
    return p_value_list
    
def split_train_val(metrics):
    keys = list(metrics.keys())
    num_elements = len(metrics[keys[0]])
    print (f"Using {num_elements} elements")
    # select a random subset of val_metrics (50% of ids)
    ids_train = np.random.choice(num_elements, num_elements//2, replace=False)
    ids_val = np.array([i for i in range(num_elements) if i not in ids_train])
    new_metrics_train = {}
    new_metrics_val = {}
    for key in keys:
        new_metrics_train[key] = np.array(metrics[key])[ids_train]
        new_metrics_val[key] = np.array(metrics[key])[ids_val]
    return new_metrics_train, new_metrics_val

def run(quant=True):
    if quant:
        gptq_model = AutoModelForCausalLM.from_pretrained(gptq_model_path, cache_dir=cache_dir, trust_remote_code=True).cuda()
        llm, tokenizer = prepare_model(gptq_model_path, cache_dir=cache_dir, quant="gptq")
    else:
        llm, tokenizer = prepare_model(model_name, cache_dir=cache_dir, quant="fp16")
    
    
    ds = load_dataset("haritzpuerto/the_pile_arxiv_50k_sample", cache_dir=cache_dir)
    
    # A_members = ds['train'].select(range(0, 10000))
    # A_nonmembers = ds['validation'].select(range(1200))
    
    # B_members = ds['train'].select(range(10000, 20000))
    # B_nonmembers = ds['validation'].select(range(1200, 2400))
    
    A_members = ds['train'].select(range(0, 1000))
    A_nonmembers = ds['validation'].select(range(1000)) 

    B_members = ds['train'].select(range(1000, 2000))
    B_nonmembers = ds['validation'].select(range(1000, 2000))
    
    metric_list = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs"]
    batch_size = 2
    
    A_members_metrics = aggregate_metrics(llm, tokenizer, A_members, metric_list, None, batch_size=batch_size)
    A_nonmembers_metrics = aggregate_metrics(llm, tokenizer, A_nonmembers, metric_list, None, batch_size=batch_size)
    
    train_metrics, val_metrics  = prepare_metrics(A_members_metrics, A_nonmembers_metrics, outliers="clip")
    
    num_samples = 2500 # How many samples to use for training and validation?
    
    np.random.shuffle(train_metrics)
    np.random.shuffle(val_metrics)
    
    # train a model by creating a train set and a held out set
    (train_x, train_y), (val_x, val_y) = get_dataset_splits(train_metrics, val_metrics, num_samples)
    
    model = train_model(train_x, train_y, num_epochs = 1000)
    
    # using the model weights, get importance of each feature, and save to csv
    weights = model.weight.data.squeeze().tolist() 
    features = list(A_members_metrics.keys())
    feature_importance = {feature: weight for feature, weight in zip(features, weights)}
    df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
    
    # auc = plot_roc_curve(model, val_x, val_y)
    
    B_members_metrics = aggregate_metrics(llm, tokenizer, B_members, metric_list, None, batch_size=batch_size)
    B_nonmembers_metrics = aggregate_metrics(llm, tokenizer, B_nonmembers, metric_list, None, batch_size=batch_size)
    
    B_members_metrics_tensor, B_nonmembers_metrics_ternsor = prepare_metrics(B_members_metrics, B_nonmembers_metrics, outliers=None, return_tensors=True)
    B_members_preds, _ = get_predictions(model, B_members_metrics_tensor, torch.tensor([0]*B_members_metrics_tensor.shape[0]))
    B_nonmembers_preds, _ = get_predictions(model, B_nonmembers_metrics_ternsor, torch.tensor([1]*B_nonmembers_metrics_ternsor.shape[0]))
    
    # p_value_list = get_p_value_list(B_members_preds, B_nonmembers_preds, list_number_samples=[1000])
    p_value_list = get_p_value_list(B_members_preds, B_nonmembers_preds, list_number_samples=list_number_samples)
    
    print(f"The null hypothesis is that the B_members scores are larger or equal than B_nonmembers.\nThe alternative hypothesis is that B_members (0) are lower than B_nonmembers (1) . The p-value is {p_value_list[-1]}")
    
    return {
        "B_members_preds": B_members_preds,
        "B_nonmembers_preds": B_nonmembers_preds,
        "p_value_list": p_value_list,
        "importance_df": df,
        "auc": auc,
        "model": model,
    }