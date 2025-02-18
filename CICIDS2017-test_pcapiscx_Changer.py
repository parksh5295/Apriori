import pyshark
import pandas as pd

cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX')
'''
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Morning.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Monday-WorkingHours.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Tuesday-WorkingHours.pcap_ISCX')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Wednesday-workingHours.pcap_ISCX')
'''
data = []

for pkt in cap:
    try:
        data.append({
            "Time": pkt.sniff_time,
            "Source": pkt.ip.src,
            "Destination": pkt.ip.dst,
            "Protocol": pkt.highest_layer,
            "Length": pkt.length
        })
    except AttributeError:
        continue  # IP 필드가 없는 패킷은 건너뜀

df = pd.DataFrame(data)
df.to_csv("../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Afternoon-DDos.csv", index=False)
'''
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Afternoon-PortScan.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Friday-WorkingHours-Morning.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Monday-WorkingHours.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Thursday-WorkingHours-Afternoon-Infilteration.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Thursday-WorkingHours-Morning-WebAttacks.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Tuesday-WorkingHours.csv')
cap = pyshark.FileCapture('../Data_Resources/CICIDS2017-test/Wednesday-workingHours.csv')
'''
