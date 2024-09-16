class Houston2013:
    input_hsi_channel = 144
    input_lidar_channel = 1
    num_class = 15
    sem_classes = []
    for i in range(num_class):
        sem_classes.append(str(i))


class MUFFL:
    input_hsi_channel = 64
    input_lidar_channel = 2
    num_class = 11
    sem_classes = []
    for i in range(num_class):
        sem_classes.append(str(i))


class Augsburg:
    input_hsi_channel = 180
    input_lidar_channel = 4
    num_class = 7
    sem_classes = []
    for i in range(num_class):
        sem_classes.append(str(i))


configs = {
    'Houston2013': Houston2013,
    'MUFFL': MUFFL,
    'Augsburg': Augsburg
}
