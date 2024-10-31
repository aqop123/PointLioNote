# This message contains essential time information of the 
# local sensor (lidar/inertial sensor) when get triggered
# 此消息包含了重要的时间信息触发本地传感器(激光雷达/惯性传感器)
Header header   # local timestamp of the trigger
uint32                 trigger_id
uint32                 event_id
time                   timestamp_host       # corresponding timestamp of the Host PC, not used