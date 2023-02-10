import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
import numpy as np

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def check_for_cuda():
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available", cuda)
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    return cuda


def print_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size=(3, 32, 32))


def draw_train_test_acc_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Train Loss")
    axs[0, 1].plot(train_acc)
    axs[0, 1].set_title("Train Accuracy")

    axs[1, 0].plot(test_losses)
    axs[1, 0].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def draw_misclassified_images(pred, target, data, main_title):
    fig = plt.figure(figsize=(20, 18))
    index = 1
    for i in range(len(target)):
        plt.subplot(5, 2, index)
        plt.axis('off')
        plt.imshow(data[i].T.squeeze(), cmap='gray_r')
        title = "Target=" + classes[target[i]] + "  Pred=" + classes[pred[i]]
        plt.gca().set_title(title)
        index += 1
    fig.suptitle(main_title, size=20)
    plt.show()


def unnormalize_and_get_rgb_images(data):
    rgb_imgs = []
    transform = transforms.ToPILImage()
    for image in data:
        new_im = image.T
        mean = np.array([0.491, 0.482, 0.446])  # mean of your dataset
        std = np.array([0.247, 0.243, 0.261])  # std of your dataset
        new_img = (std * new_im + mean) * 255
        new_img = new_img.astype("uint8")
        img = transform(new_img)
        rgb_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        rgb_imgs.append(rgb_img)
    return rgb_imgs


def draw_gradcam_vis(pred, target, grad_cam_imgs, main_title):
    fig = plt.figure(figsize=(20, 18))
    index = 1
    for i in range(len(grad_cam_imgs)):
        plt.subplot(5, 2, index)
        plt.axis('off')
        plt.imshow(grad_cam_imgs[i].squeeze(), cmap='gray_r')
        title = "Target=" + classes[target[i]] + "  Pred=" + classes[pred[i]]
        plt.gca().set_title(title)
        index += 1
    fig.suptitle(main_title, size=20)
    plt.show()


def draw_gradcam_images(model, misclass_images, pred, target):
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=1)
    rgb_imgs = unnormalize_and_get_rgb_images(misclass_images)
    grad_cam_imgs = []
    for image, rgb_img in zip(misclass_images, rgb_imgs):
        grayscale_cam = cam(input_tensor=torch.Tensor(image).unsqueeze(0).to(device))
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        grad_cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        grad_cam_imgs.append(grad_cam_img)
    draw_gradcam_vis(pred, target, grad_cam_imgs, "GradCam rep of misclassified images")
