import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton,
    QComboBox, QLabel, QHBoxLayout, QCheckBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import socket


class UDPReaderThread(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = True

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.ip, self.port))
            while self.running:
                data, _ = sock.recvfrom(1024)  # Buffer size 1024
                try:
                    message = data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    message = f"Received non-UTF-8 data: {data}"
                self.data_received.emit(message)
        except Exception as e:
            self.data_received.emit(f"UDP Error: {e}")
        finally:
            sock.close()

    def stop(self):
        self.running = False
        self.wait()


class SerialMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial/UDP Monitor")
        self.setGeometry(300, 300, 800, 600)

        self.udp_thread = None
        self.data_series = {  # Store data for each parameter
            "SPD": [],
            "ROT_SPD": [],
            "EXP_SPD": [],
            "EXP_ROT": [],
        }
        self.current_values = {  # Store current values for display
            "Current Speed": 0.0,
            "Omega": 0.0,
            "Expected Speed": 0.0,
            "Expected Omega": 0.0,
        }
        self.plot_items = {}
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        # UDP Controls for Reading
        udp_layout = QHBoxLayout()
        self.ip_label = QLabel("Read IP:")
        udp_layout.addWidget(self.ip_label)

        self.ip_input = QComboBox()
        self.ip_input.addItems(["127.0.0.1","192.168.1.131", "192.168.1.6", "192.168.1.8", "0.0.0.0"])  # Default IPs
        udp_layout.addWidget(self.ip_input)

        self.port_label = QLabel("Read Port:")
        udp_layout.addWidget(self.port_label)

        self.port_input = QComboBox()
        self.port_input.addItems(["5005", "3333"])  # Default ports
        udp_layout.addWidget(self.port_input)

        main_layout.addLayout(udp_layout)

        # UDP Controls for Sending
        send_layout = QHBoxLayout()
        self.send_ip_label = QLabel("Send IP:")
        send_layout.addWidget(self.send_ip_label)

        self.send_ip_input = QLineEdit("127.0.0.1")
        send_layout.addWidget(self.send_ip_input)

        self.send_port_label = QLabel("Send Port:")
        send_layout.addWidget(self.send_port_label)

        self.send_port_input = QLineEdit("5006")
        send_layout.addWidget(self.send_port_input)

        main_layout.addLayout(send_layout)

        # Connect and Stop buttons
        controls_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_udp)
        controls_layout.addWidget(self.connect_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_reading)
        controls_layout.addWidget(self.stop_button)
        main_layout.addLayout(controls_layout)

        # Real-time value display
        value_layout = QVBoxLayout()
        self.value_labels = {}
        for key in self.current_values:
            label = QLabel(f"{key}: 0.0")
            self.value_labels[key] = label
            value_layout.addWidget(label)
        main_layout.addLayout(value_layout)

        # Parameter input fields and buttons
        param_layout = QVBoxLayout()
        self.params = {}
        for param in ["FWD_KP", "FWD_KD", "FWD_KI", "ROT_KP", "ROT_KD", "ROT_KI"]:
            row_layout = QHBoxLayout()
            label = QLabel(f"{param}:")
            input_field = QLineEdit()
            input_field.setPlaceholderText("Enter value")
            button = QPushButton("Send")
            button.clicked.connect(lambda _, p=param, i=input_field: self.send_param_udp(p, i))
            row_layout.addWidget(label)
            row_layout.addWidget(input_field)
            row_layout.addWidget(button)
            param_layout.addLayout(row_layout)
            self.params[param] = input_field
        main_layout.addLayout(param_layout)

        # Output and plot widgets
        self.init_output_and_plot(main_layout)

        main_widget.setLayout(main_layout)

    def init_output_and_plot(self, layout):
        # Output area
        self.other_output = QTextEdit()
        self.other_output.setReadOnly(True)
        self.other_output.setPlaceholderText("Other Messages")
        layout.addWidget(QLabel("Other Messages:"))
        layout.addWidget(self.other_output)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Real-Time Data Plot", color='b', size="12pt")
        self.plot_widget.setLabel("left", "Value", color='blue')
        self.plot_widget.setLabel("bottom", "Sample", color='blue')
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(QLabel("Real-Time Plot:"))
        layout.addWidget(self.plot_widget)

        # Header selection checkboxes
        header_layout = QHBoxLayout()
        self.header_checkboxes = {}
        colors = {"SPD": "r", "ROT_SPD": "g", "EXP_SPD": "b", "EXP_ROT": "m"}
        for header, color in colors.items():
            checkbox = QCheckBox(header)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_plot_visibility)
            self.header_checkboxes[header] = checkbox
            self.plot_items[header] = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2), name=header
            )
            header_layout.addWidget(checkbox)
        layout.addLayout(header_layout)

    def connect_to_udp(self):
        ip = self.ip_input.currentText()
        port = int(self.port_input.currentText())

        try:
            self.udp_thread = UDPReaderThread(ip, port)
            self.udp_thread.data_received.connect(self.update_output)
            self.udp_thread.start()

            self.other_output.append(f"Listening on {ip}:{port} for UDP messages.")
            self.connect_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        except Exception as e:
            self.other_output.append(f"UDP Connection Error: {e}")

    def stop_reading(self):
        if self.udp_thread:
            self.udp_thread.stop()
            self.udp_thread = None
        self.other_output.append("Stopped reading.")
        self.connect_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_output(self, data):
        if data.startswith("VELOCITY:"):
            try:
                parts = data.split(":")[1].split(",")
                exp_spd, spd, exp_rot, rot_spd = map(float, parts)

                # Update data series
                self.data_series["EXP_SPD"].append(exp_spd)
                self.data_series["SPD"].append(spd)
                self.data_series["EXP_ROT"].append(exp_rot)
                self.data_series["ROT_SPD"].append(rot_spd)

                # Trim series to avoid memory overflow
                for key in self.data_series:
                    if len(self.data_series[key]) > 2500:
                        self.data_series[key].pop(0)

                # Update real-time display
                self.current_values["Expected Speed"] = exp_spd
                self.value_labels["Expected Speed"].setText(f"Expected Speed: {exp_spd}")
                self.current_values["Current Speed"] = spd
                self.value_labels["Current Speed"].setText(f"Current Speed: {spd}")
                self.current_values["Expected Omega"] = exp_rot
                self.value_labels["Expected Omega"].setText(f"Expected Omega: {exp_rot}")
                self.current_values["Omega"] = rot_spd
                self.value_labels["Omega"].setText(f"Omega: {rot_spd}")

                # Update plots
                for header in self.data_series:
                    self.update_plot(header)
            except (IndexError, ValueError):
                self.other_output.append(f"Invalid VELOCITY format: {data}")
        else:
            self.other_output.append(data)

    def update_plot(self, header):
        if self.header_checkboxes[header].isChecked():
            self.plot_items[header].setData(self.data_series[header])
        else:
            self.plot_items[header].clear()

    def update_plot_visibility(self):
        for header, checkbox in self.header_checkboxes.items():
            if not checkbox.isChecked():
                self.plot_items[header].clear()

    def send_param_udp(self, param, input_field):
        value = input_field.text()
        if value:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                send_ip = self.send_ip_input.text()
                send_port = int(self.send_port_input.text())
                message = f"{param}={value}"
                sock.sendto(message.encode('utf-8'), (send_ip, send_port))
                self.other_output.append(f"Sent: {message} to {send_ip}:{send_port}")
            except Exception as e:
                self.other_output.append(f"Error sending {param}: {e}")
        else:
            self.other_output.append(f"Error: No value entered for {param}")


def main():
    app = QApplication(sys.argv)
    window = SerialMonitor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
