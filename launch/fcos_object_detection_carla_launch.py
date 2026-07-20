from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory('fcos_object_detection')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (CARLA) clock if true'
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(pkg_share, 'launch', 'fcos_object_detection_rviz_launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'input_topic': '/carla/hero/cam2/image',  # CARLA-specific default
            'rviz_config': join(pkg_share, 'rviz', 'fcos_object_detection_carla.rviz'),
        }.items()
    )

    return LaunchDescription([
        declare_use_sim_time,
        rviz_launch,
    ])
