import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def to_one_hot(labels):
    """Convert integer labels to one-hot encoding; returns one-hot array."""
    num_classes = len(set(labels))
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels - 1] = 1
    return one_hot

def softmax(z):
    """Compute row-wise softmax; returns probability array."""
    # BEGIN YOUR CODE HERE (~1-3 lines)
    
    
    z = z- z.max(axis=1, keepdims=True)
    z = np.exp(z)
    proba = z /z.sum(axis=1, keepdims =True)

    # END YOUR CODE HERE
    return proba


def cross_entropy_loss(W, b, images, labels):
    """Compute cross-entropy loss for predictions; returns scalar loss."""
    # BEGIN YOUR CODE HERE (~2-6 lines)
    logits =  images@W + b
    proba = softmax(logits)

    class_index = np.argmax(labels,axis=1)
    prob_true = proba[np.arange(images.shape[0]), class_index]

    keep_above_zero = 1e-6


    loss = -np.mean(np.log(prob_true+keep_above_zero))
    loss += 0.5 * alpha * np.sum(W ** 2) # here L2 reg on the weights only


    # END YOUR CODE HERE



    return loss

def compute_gradient(W, b, images, labels, alpha):
    """Compute gradients w.r.t. weights and bias; returns (dW, db)."""
    # BEGIN YOUR CODE HERE (~4-7 lines)
    logits =  images@W + b
    proba = softmax(logits)

    B=images.shape[0]
    Delta = (proba - labels)/B

    dW = images.T @ Delta
    dW += alpha * W # this applies L2 in the gradient
    
    db = Delta.sum(axis=0)

    # END YOUR CODE HERE
    return dW, db

def compute_accuracy(W, b, images, labels):
    """Compute classification accuracy; returns fraction correct."""
    # BEGIN YOUR CODE HERE (~2-5 lines)

    logits =  images@W + b
    proba = softmax(logits)

    prediction = np.argmax(proba, axis=1)
    output = np.argmax(labels,axis=1)

    acc = np.mean(prediction == output)
    # END YOUR CODE HERE
    return acc
            
def show_weights(W):
    """Render weights as image patches; returns None."""
    img_size = int(W.shape[0] ** 0.5)
    canvas = np.zeros((img_size, img_size * W.shape[1]))
    for idx, col in enumerate(range(0, canvas.shape[1], img_size)):
        canvas[:, col:col+img_size] = np.reshape(W[:-1, idx], (img_size, img_size))
    plt.imshow(canvas, cmap='gray')
    plt.show()

def train_softmax_classifier(train_images, train_labels, val_images, val_labels,
                             learning_rate=1e-5, batch_size=16, nepochs=100, alpha=0.0):
    """Train softmax classifier; returns (W, b, loss, accuracy)."""
    num_batches = train_images.shape[0] // batch_size
    n_features = train_images.shape[1]
    n_classes = train_labels.shape[1]

    # Initialize weights and bias
    # BEGIN YOUR CODE HERE (~2 lines)
    W = 0.01 * np.random.randn(n_features, n_classes)
    b = np.zeros(n_classes)
    # END YOUR CODE HERE

    cost, acc = float('inf'), 0.0
    for epoch in range(nepochs):
        np.random.seed(epoch)
        perm = np.random.permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        for batch in range(num_batches):
            # what do you need to do in each iteration?
            # BEGIN YOUR CODE HERE (~5 lines)

            s = batch * batch_size
            e = s + batch_size

            xb = train_images[s:e]
            yb = train_labels[s:e]


            dW, db = compute_gradient(W, b, xb, yb, alpha)
            
            W -= learning_rate * dW
            b -= learning_rate * db


            # END YOUR CODE HERE
        # what do you need to compute at the end of each epoch?
        # BEGIN YOUR CODE HERE (~3 lines)
        cost = cross_entropy_loss(W, b, val_images, val_labels)
        acc  = compute_accuracy(W, b, val_images, val_labels)
        # END YOUR CODE HERE
    return W, b, cost, acc

# def add_bias(images):
#     return np.hstack((images, np.ones((images.shape[0], 1))))

if __name__ == "__main__":
    # Load data
    np.random.seed(541) 
    
    training_images = np.load("fashion_mnist_train_images.npy") / 255.0 - 0.5
    training_labels = to_one_hot(np.load("fashion_mnist_train_labels.npy"))
    testing_images = np.load("fashion_mnist_test_images.npy") / 255.0 - 0.5
    testing_labels = to_one_hot(np.load("fashion_mnist_test_labels.npy"))

    # split training into training and validation using train_test_split
    x_train, x_val, y_train, y_val = train_test_split(training_images, training_labels,
                                                      test_size=0.2, random_state=541)

    # List hyperparameters to try
    # BEGIN YOUR CODE HERE (~2-3 lines)
    learn_rs =[0.1,0.01,0.001]
    batch_szs = [8,16,32,64,128]
    alpha_values = [0.0, 0.001, 0.01]
    # END YOUR CODE HERE

    # Initialize varjables to keep track of best hyperparameters
    # BEGIN YOUR CODE HERE (~3 lines)
    best_accuracy = -1.0 
    best_alpha = 0
    best_lr = 0
    best_bs = 0
    best_W = 0
    best_b = 0

    # END YOUR CODE HERE

    # Train model
    # BEGIN YOUR CODE HERE (~7-10 lines)

    # Train model (grid over all unique combos)
    run_idx = 0
    total = len(learn_rs) * len(batch_szs) * len(alpha_values)
    
    for lnr in learn_rs:
        for bts in batch_szs:
            for alpha in alpha_values:
                run_idx += 1
                print(f"Starting combo {run_idx}/{total}: lr={lnr}, batch_size={bts}, alpha={alpha}", flush=True)
                
                w, bias, loss, accuracy = train_softmax_classifier(
                    x_train, y_train, x_val, y_val,
                    learning_rate=lnr, batch_size=bts, nepochs=100, alpha=alpha
                )
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_lr = lnr
                    best_bs = bts
                    best_alpha = alpha
                    best_W = w
                    best_b = bias
                    print(f" -> new best acc={best_accuracy:.4f}", flush=True)
                    
    print("Tested all unique hyperparameter combinations.")
    
    # END YOUR CODE HERE

    # Retrain model on full training set with best hyperparameters and evaluate on test set
    # BEGIN YOUR CODE HERE (~1 line)
    final_W, final_b, loss, acc = train_softmax_classifier(
        training_images, training_labels, testing_images, testing_labels,
        learning_rate=best_lr, batch_size=best_bs, nepochs=100, alpha=best_alpha)
    
    test_loss = cross_entropy_loss(final_W, final_b, testing_images, testing_labels, best_alpha)
    test_accuracy = compute_accuracy(final_W, final_b, testing_images, testing_labels)

    
    # END YOUR CODE HERE

    # showWeights(lr[:,:])
    
    
