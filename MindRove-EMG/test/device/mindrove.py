from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
import logging

def print_channel_indices(board_shim):
    """Function to iterate over get_<type>_channels and print channel indices by type."""
    board_id = BoardIds.MINDROVE_WIFI_BOARD.value

    # Get various channel types
    try:
        eeg_channels = board_shim.get_eeg_channels(board_id)
        print(f"EEG Channels: {eeg_channels}")
    except Exception as e:
        print(f"EEG Channels: Not supported ({str(e)})")
    
    try:
        emg_channels = board_shim.get_emg_channels(board_id)
        print(f"EMG Channels: {emg_channels}")
    except Exception as e:
        print(f"EMG Channels: Not supported ({str(e)})")

    try:
        counter_idx = board_shim.get_package_num_channel(board_id)
        print(f"Counter Channel: {counter_idx}")
    except Exception as e:
        print(f"Counter not supported. ({str(e)})")

    try:
        trigger_idx = board_shim.get_other_channels(board_id)[0]
        print(f"Trigger Channel (Beep/Boop): {trigger_idx}")
    except Exception as e:
        print(f"Triggers not supported. ({str(e)})")

    try:
        battery_idx = board_shim.get_battery_channel(board_id)
        print(f"Battery Channel: {battery_idx}")
    except Exception as e:
        print(f"Battery channel not supported. ({str(e)})")

    try:
        marker_idx = board_shim.get_marker_channel(board_id)
        print(f"Marker Channel: {marker_idx}")
    except Exception as e:
        print(f"Marker channel not supported. ({str(e)})")

    try:
        ecg_channels = board_shim.get_ecg_channels(board_id)
        print(f"ECG Channels: {ecg_channels}")
    except Exception as e:
        print(f"ECG Channels: Not supported ({str(e)})")

    try:
        ppg_channels = board_shim.get_ppg_channels(board_id)
        print(f"PPG Channels: {ppg_channels}")
    except Exception as e:
        print(f"PPG Channels: Not supported ({str(e)})")

    try:
        accel_channels = board_shim.get_accel_channels(board_id)
        print(f"Accelerometer Channels: {accel_channels}")
    except Exception as e:
        print(f"Accelerometer Channels: Not supported ({str(e)})")

    try:
        gyro_channels = board_shim.get_gyro_channels(board_id)
        print(f"Gyroscope Channels: {gyro_channels}")
    except Exception as e:
        print(f"Gyroscope Channels: Not supported ({str(e)})")

    try:
        eda_channels = board_shim.get_eda_channels(board_id)
        print(f"EDA Channels: {eda_channels}")
    except Exception as e:
        print(f"EDA Channels: Not supported ({str(e)})")

    try:
        temperature_channels = board_shim.get_temperature_channels(board_id)
        print(f"Temperature Channels: {temperature_channels}")
    except Exception as e:
        print(f"Temperature Channels: Not supported ({str(e)})")

    try:
        resistance_channels = board_shim.get_resistance_channels(board_id)
        print(f"Resistance Channels: {resistance_channels}")
    except Exception as e:
        print(f"Resistance Channels: Not supported ({str(e)})")

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    # Set up the MindRoveInputParams for MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()

    board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

    try:
        # Prepare and start the session
        board_shim.prepare_session()
        board_shim.start_stream()

        # Print channel indices
        print_channel_indices(board_shim)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
    
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == "__main__":
    main()
