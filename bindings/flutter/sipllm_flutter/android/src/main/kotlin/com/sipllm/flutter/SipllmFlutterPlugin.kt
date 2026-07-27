package com.sipllm.flutter

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

/**
 * Entry point registered by the Flutter FFI plugin machinery (see
 * `pubspec.yaml` -> plugin.platforms.android.pluginClass).
 *
 * Wires up three channels:
 *  - MethodChannel `sipllm/device` — one-shot [DeviceProfiler.profile] lookup.
 *  - MethodChannel `sipllm/wear`   — Wear OS model-transfer control surface.
 *  - EventChannel  `sipllm/wear/events` — streamed [WearModelTransfer] progress.
 *
 * The native engine library (libsipllm_ffi.so) is loaded by the Dart FFI layer
 * directly; this plugin does not touch it — it only owns the platform-channel
 * bridges that FFI cannot express (device metadata + Data Layer transfer).
 */
class SipllmFlutterPlugin : FlutterPlugin {

    private var deviceChannel: MethodChannel? = null
    private var wearChannel: MethodChannel? = null
    private var eventChannel: EventChannel? = null
    private var transfer: WearModelTransfer? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        val context = binding.applicationContext
        val messenger = binding.binaryMessenger

        val device = MethodChannel(messenger, "sipllm/device")
        device.setMethodCallHandler { call, result ->
            when (call.method) {
                "getDeviceProfile" -> result.success(DeviceProfiler.profile(context))
                else -> result.notImplemented()
            }
        }
        deviceChannel = device

        val xfer = WearModelTransfer(context)
        transfer = xfer

        val events = EventChannel(messenger, "sipllm/wear/events")
        events.setStreamHandler(xfer.streamHandler)
        eventChannel = events

        val wear = MethodChannel(messenger, "sipllm/wear")
        wear.setMethodCallHandler { call, result -> xfer.handle(call, result) }
        wearChannel = wear
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        deviceChannel?.setMethodCallHandler(null)
        wearChannel?.setMethodCallHandler(null)
        eventChannel?.setStreamHandler(null)
        transfer?.dispose()

        deviceChannel = null
        wearChannel = null
        eventChannel = null
        transfer = null
    }
}
