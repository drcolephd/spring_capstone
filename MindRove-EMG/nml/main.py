import sys, argparse
from PyQt5.QtWidgets import QApplication
from nml.application import Application

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Filename to use when saving the data. Default: default', required=False, default='default')
    parser.add_argument('--synth', type=int, help='Set to 1 to generate synthetic data. Default: 0.', required=False, default=0)
    parser.add_argument('--suffix', type=int, help='The suffix integer added to end of log file names. Default: 0.', required=False, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    main_app = Application(app, args)
    main_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()