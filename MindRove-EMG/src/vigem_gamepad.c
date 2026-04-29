#include <windows.h>
#include <ViGEm/Client.h>
#include <stdint.h>
#include <stdbool.h>

static PVIGEM_CLIENT client = NULL;
static PVIGEM_TARGET target = NULL;
static HWND targetWindow = NULL; // Store the HWND for the target window

#define WM_GAMEPAD_INPUT (WM_USER + 1) // Custom message ID for gamepad input

__declspec(dllexport) int initialize_gamepad() {
    client = vigem_alloc();
    if (client == NULL || !VIGEM_SUCCESS(vigem_connect(client))) {
        return -1; // Error initializing client
    }

    target = vigem_target_x360_alloc();
    if (!VIGEM_SUCCESS(vigem_target_add(client, target))) {
        return -2; // Error adding controller
    }

    return 0; // Success
}

__declspec(dllexport) int set_target_window(HWND hwnd) {
    if (hwnd == NULL) {
        OutputDebugStringA("set_target_window: Invalid HWND\n");
        return -1; // Invalid HWND
    }
    targetWindow = hwnd;
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "set_target_window: HWND set to %p\n", hwnd);
    OutputDebugStringA(buffer);
    return 0; // Success
}


__declspec(dllexport) int send_gamepad_input(uint16_t buttons, int8_t leftX, int8_t leftY, int8_t rightX, int8_t rightY) {
    if (!client || !target) return -3; // Not initialized

    // Ensure the target window has focus
    if (targetWindow) {
        BOOL focusResult = SetForegroundWindow(targetWindow);
        if (!focusResult) {
            OutputDebugStringA("send_gamepad_input: Failed to set foreground window.\n");
        }
    }
    XUSB_REPORT report = { 0 };
    report.wButtons = buttons;
    report.bLeftTrigger = 0; // No trigger pressed
    report.bRightTrigger = 0;
    report.sThumbLX = leftX * 256;
    report.sThumbLY = leftY * 256;
    report.sThumbRX = rightX * 256;
    report.sThumbRY = rightY * 256;

    if (!VIGEM_SUCCESS(vigem_target_x360_update(client, target, report))) {
        return -4; // Error sending input
    }

    return 0; // Success
}

__declspec(dllexport) void cleanup_gamepad() {
    if (target) {
        vigem_target_remove(client, target);
        vigem_target_free(target);
    }
    if (client) {
        vigem_disconnect(client);
        vigem_free(client);
    }
    client = NULL;
    target = NULL;
    targetWindow = NULL;
}
