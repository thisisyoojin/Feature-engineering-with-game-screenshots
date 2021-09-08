import numpy as np


def pk(actual, pred, k=3):
    """
    Precision at k(P@k):
    the number of correct prdictions considering only top k elements of each class divided by k

    params
    ======
    actual(tensor): actual targets from the dataset(y_batch)
    pred(tensor): logits predicted from the model
    k(int): how many predictions to include for precision
    """

    # retun 0 if k is 0
    if k == 0:
        return 0    

    k_pred = [d[0] for d in sorted(enumerate(pred), key=lambda a:a[1], reverse=True)[:k]]
    actual = [i for i, a in enumerate(actual) if a == 1]

    hit = 0

    for a in actual:
        for p in k_pred:
            if a == p:
                hit += 1

    return hit/len(k_pred)



def apk(actual, pred, k=3):
    """
    Average Precision at K(APK):
    the average of all precision at k from (1 to k)

    params
    ======
    actual(tensor): actual targets from the dataset(y_batch)
    pred(tensor): logits predicted from the model
    k(int): how many predictions to include for precision
    """
    precision = []
    
    for i in range(1, k+1):
        precision.append(pk(actual, pred, i))
    
    if len(precision) == 0:
        return 0

    return np.mean(precision)


def mapk(actual, pred, k=3):
    """
    Mean Average Precision

    params
    ======
    actual(tensor): actual targets from the dataset(y_batch)
    pred(tensor): logits predicted from the model
    k(int): how many predictions to include for precision
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, pred)])



def best_mapk(loader, k):
    """
    Mean Average Precision

    params
    ======
    loader(Dataloader): dataloader with dataset to calculate best mapk
    k(int): how many predictions to include for precision
    """
    eval_score = 0
    
    for _, y_batch in loader:
        eval_score += mapk(y_batch, y_batch, k)
    
    return eval_score/len(loader.dataset)