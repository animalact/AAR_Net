import time

import torch.optim as optim
from torch.utils.data import DataLoader

def test(model1, model2, testloader,epoch):
    model1.load_state_dict(torch.load('output/weights/save_param1_' + str(epoch)+ '.pth'))
    model2.load_state_dict(torch.load('output/weights/save_param2_' + str(epoch)+ '.pth'))

    model1.eval()
    model2.eval()

    shape = None
    correct = None
    total = None
    for imgs, labels in testloader:
        imgs, labels = torch.Tensor(imgs.float()).cuda(), torch.Tensor(labels.float()).cuda()

        features = model1(imgs)
        out = model2(features)
        if shape is None:
            shape = out.shape
            correct = torch.zeros(shape[1])
            total = torch.zeros(shape[1])

        total[torch.argmax(labels,1)] += 1
        if torch.argmax(out, 1) == torch.argmax(labels, 1):
            correct[torch.argmax(labels)] += 1


    print()
    print("-"*10)
    print(correct/total)
    print(torch.sum(correct)/torch.sum(total))


def train(model1, model2, trainloader, testloader):

    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    loss_fn = nn.NLLLoss()
    tot_len = int(len(trainloader.dataset)//trainloader.batch_size+1)

    prev = time.time()

    for epoch in range(50):
        print(f"epoch: {epoch} is started")
        print()
        model1.train()
        model2.train()
        cur_id = 0
        for imgs, labels in trainloader:
            cur_id+=1
            prev = time.time()
            imgs, labels = torch.Tensor(imgs.float()).cuda(),torch.Tensor(labels.float()).cuda()
            # imgs -> (None, ch, frame, w, h) -> (batch, 3, 30, 256, 256)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            features = model1(imgs)
            out = model2(features)

            loss = loss_fn(out,torch.max(labels, 1)[1].long())
            loss.backward()

            optimizer1.step()
            optimizer2.step()
            print("", end="\r")
            print(f"{cur_id} / {tot_len} batch done : {round(time.time()-prev,2)}s ", end="")

        torch.save(model1.state_dict(), 'output/weights/save_param1_' + str(epoch)+ '.pth')
        torch.save(model2.state_dict(), 'output/weights/save_param2_' + str(epoch)+ '.pth')
        test(model1,model2,testloader,epoch=epoch)

        print("epoch:", epoch," _ loss:", loss.cpu().detach().numpy())


if __name__ == "__main__":
    from lib.models.AARNet_LSTM import *
    from lib.dataset.aar_lstm_dataset import AARDataset
    select = [2,7,12]
    train_dataset = AARDataset(data_path="./data/", category="cat", anno_path='./data/train', frame_thr=30, skip=15, select=select)
    test_dataset = AARDataset(data_path="./data/", category="cat", anno_path='./data/test', frame_thr=30, skip=1000, select=select)

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)
    #
    net_cnn = CNN().cuda()
    # net_cnn = CNN3D().cuda()
    net_wellknown = Net(name="resnet18").cuda()
    net_lstm = LSTM(len(select)).cuda()
    #
    train(net_wellknown, net_lstm, train_loader, test_loader)

