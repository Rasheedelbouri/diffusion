from model import UNet


if __name__ == '__main__':
    nodes = [100, 250, 500, 1000]
    model = UNet(nodes)
