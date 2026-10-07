import serial
import struct
import math

import serial.tools.list_ports

def find_port(vid=0x1ffb, pid=0x00bb):
    for p in serial.tools.list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    return None

PORT = find_port() or "/dev/ttyACM1"
#PORT = "/dev/ttyACM0"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
print("Opened:", ser.is_open, ser.name)

def u16_be(b): return (b[0] << 8) | b[1]
def i16_be(b): return struct.unpack(">h", b)[0]
def f32_be(b): return struct.unpack(">f", b)[0]
def checksum(pkt): return sum(pkt) & 0xFFFF

def data_len_from_pt(pt):
    has_data = (pt >> 7) & 1
    is_batch = (pt >> 6) & 1
    bl = (pt >> 2) & 0x0F
    if not has_data:
        return 0
    return (4 * bl) if is_batch else 4

def is_all_zero(vals, eps=1e-12):
    return all(abs(v) < eps for v in vals)

def parse_euler_0x70(data):
    # 5 registers from 0x70..0x74, each 4 bytes
    # 0x70: phi(roll), theta(pitch) as int16/int16 scaled by /91.02222
    # 0x71: psi(yaw) in upper int16 scaled /91.02222
    # 0x72: roll_rate, pitch_rate int16/int16 scaled /16
    # 0x73: yaw_rate in upper int16 scaled /16
    # 0x74: time float
    phi_raw = i16_be(data[0:2])
    theta_raw = i16_be(data[2:4])

    psi_raw = i16_be(data[4:6])

    roll_rate_raw = i16_be(data[8:10])
    pitch_rate_raw = i16_be(data[10:12])

    yaw_rate_raw = i16_be(data[12:14])

    t = f32_be(data[16:20])

    phi = phi_raw / 91.02222
    theta = theta_raw / 91.02222
    psi = psi_raw / 91.02222

    roll_rate = roll_rate_raw / 16.0
    pitch_rate = pitch_rate_raw / 16.0
    yaw_rate = yaw_rate_raw / 16.0

    return phi, theta, psi, roll_rate, pitch_rate, yaw_rate, t

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print(f"Listening on {PORT} @ {BAUD} ...")

    while True:
        if ser.read(1) != b's': continue
        if ser.read(1) != b'n': continue
        if ser.read(1) != b'p': continue

        hdr = ser.read(2)
        if len(hdr) < 2: continue
        pt, addr = hdr[0], hdr[1]

        dlen = data_len_from_pt(pt)
        data = ser.read(dlen)
        cks = ser.read(2)
        if len(data) != dlen or len(cks) != 2: continue

        pkt_wo_cks = b"snp" + bytes([pt, addr]) + data
        if u16_be(cks) != checksum(pkt_wo_cks):
            continue

        # 0x61..0x6C (12 float regs): gyro/accel/mag processed + times
        if pt == 0xF0 and addr == 0x61 and dlen == 48:
            vals = [f32_be(data[i:i+4]) for i in range(0, 48, 4)]
            gx, gy, gz, t_g = vals[0], vals[1], vals[2], vals[3]
            ax, ay, az, t_a = vals[4], vals[5], vals[6], vals[7]
            mx, my, mz, t_m = vals[8], vals[9], vals[10], vals[11]

            if is_all_zero([gx, gy, gz, ax, ay, az, mx, my, mz]):
                continue

            print(
                f"gyro[dps]=({gx:8.3f},{gy:8.3f},{gz:8.3f})  "
                f"acc[m/s^2]=({ax:7.3f},{ay:7.3f},{az:7.3f})  "
                f"mag=({mx:7.3f},{my:7.3f},{mz:7.3f})"
            )

        # 0x70..0x74 (Euler + rates + time)
        elif pt == 0xD4 and addr == 0x70 and dlen == 20:
            roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, t = parse_euler_0x70(data)
            # optional: skip clearly invalid startup zeros
            if is_all_zero([roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]):
                continue

            print(
                f"euler[deg]=({roll:7.2f},{pitch:7.2f},{yaw:7.2f})  "
                f"rates[dps]=({roll_rate:7.2f},{pitch_rate:7.2f},{yaw_rate:7.2f})"
            )

        # 0x89..0x8B gyro bias estimate
        elif pt == 0xCC and addr == 0x89 and dlen == 12:
            bx = f32_be(data[0:4])
            by = f32_be(data[4:8])
            bz = f32_be(data[8:12])
            print(f"gyro_bias[dps]=({bx:7.4f},{by:7.4f},{bz:7.4f})")