from ..ml_utils.summary_writer import SummaryWriter
from ..ml_utils.misc import CurrentDir
curdir = CurrentDir(__file__)

def main():
    writer = SummaryWriter(curdir('runs'))
    writer.plot(curdir('imgs'))


if __name__ == '__main__':
    main()
