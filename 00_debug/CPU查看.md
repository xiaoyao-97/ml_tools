



# windows查看CPU数量
wmic cpu get NumberOfCores,NumberOfLogicalProcessors


# macbook查看CPU数量
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu


# linux查看CPU数量
lscpu | grep '^CPU(s):'
grep -c ^processor /proc/cpuinfo



# windows查看GPU
1. 任务管理器-性能
2. dxdiag
3. wmic path win32_VideoController get name

# macbook查看GPU
system_profiler SPDisplaysDataType


# linux查看GPU
lspci | grep -i vga

# 性能监控

## windows
1. 任务管理器：
按 Ctrl + Shift + Esc 打开任务管理器。
切换到“性能”选项卡，可以查看CPU、内存、磁盘和网络的使用情况。选择“GPU”可以查看GPU的使用情况。

2. 资源监视器：
按 Win + R 打开“运行”对话框，输入 resmon 并按回车。
资源监视器可以提供更详细的CPU、内存、磁盘和网络使用情况。

3. 性能监视器：
按 Win + R 打开“运行”对话框，输入 perfmon 并按回车。
性能监视器允许你创建自定义的性能数据集，并且可以监控各种系统资源的使用情况。

4. 第三方工具：
工具如HWMonitor、GPU-Z、MSI Afterburner等可以提供更详细的硬件使用情况，包括温度、电压等。

## mac

活动监视器：

可以在“应用程序”->“实用工具”中找到“活动监视器”。
通过“CPU”、“内存”、“能源”、“磁盘”和“网络”选项卡，可以监控相应资源的使用情况。


## linux
1. 命令行工具：
top 和 htop：查看实时的CPU、内存和进程信息。htop 提供了一个更友好的用户界面。
free -h：查看内存使用情况。
iostat：查看磁盘I/O使用情况。
vmstat：查看系统的总体性能信息。
nvidia-smi：如果使用NVIDIA显卡，可以用来查看GPU的使用情况。
2. GUI工具：
GNOME系统监视器：在GNOME桌面环境中可以使用，提供图形界面来监控系统资源。
KDE系统监视器：在KDE桌面环境中可以使用，提供类似的功能。
3. 第三方工具：
工具如Netdata、Glances等可以提供全面的系统监控，并且支持通过网页浏览器查看。





