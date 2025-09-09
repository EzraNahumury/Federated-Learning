import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff



#============================================== 1. Preparing The Input Data ==============================================
from matplotlib import pyplot as plt

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# print(len(emnist_train.client_ids))
# print(emnist_train.element_type_structure)

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]
)
example_element = next(iter(example_dataset))
# print("label : ", example_element['label'].numpy())


plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
# plt.savefig("sample.png")






# ============================================== 2. Exploring heterogeneity in federated data ==============================================

#Example MNIST digits for one client
figure = plt.figure(figsize=(20,4))
j = 0

for example in example_dataset.take(40):
    plt.subplot(4, 10, j+1)
    plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
    plt.axis('off')
    j += 1
    # plt.savefig("Example.png")



#Number of examples per layer for a sample of clients
f = plt.figure(figsize=(12,7))
f.suptitle('Label Counts for a sample of clients')
for i in range(6):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[1]
    )
    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        #Append counts individually per label to make plots
        #more colorful instead of one color per plot
        label = example['label'].numpy()
        plot_data['label'].append(label)
    plt.subplot(2,3, i+1)
    plt.title('Client {}'.format(i))
    for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0,1,2,3,4,5,6,7,8,9,10]
            )
# plt.savefig("Client.png")


# Each client has different mean images, meaning each client  will be nudging
# the model in their own directions locally

for i in range (5):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[i]
    )
    plot_data = collections.defaultdict(list)
    for example in client_dataset :
        plot_data[example['label'].numpy()].append(example['pixels'].numpy())
    f = plt.figure(figsize=(12, 5))
    f.suptitle(f"Client #{i}'s Mean Image Per label")
    for j in range(10):
        mean_img = np.mean(plot_data[j],0)
        plt.subplot(2,5, j+1)
        plt.imshow(mean_img.reshape((28,28)))
        plt.axis('off')
    # plt.savefig(f"Client_{i}.png")
    plt.close() 



# ============================================== 3. Preporcessing the input data  ==============================================

NUM_CLIENTS = 20
NUM_EPOCHS = 15
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            x = tf.reshape(element['pixels'], [-1, 784]),
            y = tf.reshape(element['label'], [-1,1])
        )
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))
# print(sample_batch)


def make_federated_data(client_data, client_ids) :
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
# print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
# print('First dataset: {d}'.format(d=federated_train_data[0]))




# ============================================== 4. Creating a model with Keras ==============================================

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])



def model_fn():

     # We _must_ create a new model here, and _not_ capture it from an external
     # scope. TFF will call this within different graph contexts.
     keras_model = create_keras_model()
     return tff.learning.models.from_keras_model(
        keras_model,
        input_spec = preprocessed_example_dataset.element_spec,
        loss  = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
     )





# ============================================== 5. Training the model on federated data ==============================================
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02))

train_state = training_process.initialize()

# Koleksi metric per round
train_acc_hist, train_loss_hist, rounds_hist = [], [], []

NUM_ROUNDS = 11  # akan menghasilkan 11 titik (1..11)
for rnd in range(1, NUM_ROUNDS + 1):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state

    mt = result.metrics['client_work']['train']
    acc = float(mt.get('sparse_categorical_accuracy', 0.0))  # <- perbaiki ejaan
    loss = float(mt.get('loss', 0.0))

    train_acc_hist.append(acc)
    train_loss_hist.append(loss)
    rounds_hist.append(rnd)  # <- pakai nomor round, bukan 1

    print(f"round {rnd:2d} | acc={acc:.4f} | loss={loss:.4f}")

# ------------------ Plot ------------------
# Accuracy per round
plt.figure()
plt.plot(rounds_hist, train_acc_hist, linewidth=2, marker='o', label='Train Accuracy')
plt.title('Accuracy per Round')
plt.xlabel('Round'); plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_per_round.png', dpi=150)

# Loss per round (aktifkan log kalau perlu)
plt.figure()
plt.plot(rounds_hist, train_loss_hist, linewidth=2, marker='o', label='Train Loss')
# plt.yscale('log')  # nyalakan jika loss turun tajam
plt.title('Loss per Round')
plt.xlabel('Round'); plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('loss_per_round.png', dpi=150)


# --- Loss (log-scale) ---
# Aman-kan nilai <= 0 sebelum log
loss_arr = np.array(train_loss_hist, dtype=float)
eps = 1e-12
if np.any(loss_arr <= 0):
    print("Catatan: ada nilai loss <= 0, ditambahkan epsilon agar bisa log-scale.")
loss_log = np.clip(loss_arr, eps, None)

# Cara 1: langsung semilogy
plt.figure()
plt.semilogy(rounds_hist, loss_log, linewidth=2, marker='o', label='Train Loss')
plt.title('Loss per Round (Log Scale)')
plt.xlabel('Round'); plt.ylabel('Loss (log)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(); plt.tight_layout()
plt.savefig('loss_per_round_log.png', dpi=150)