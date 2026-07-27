package com.sipllm.flutter

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build

/**
 * Reports static build/ABI/device capabilities to the Dart side. These need no
 * model and are cheap to gather; the Dart [ArchDetector] consumes the map over
 * the `sipllm/device` MethodChannel.
 */
internal object DeviceProfiler {

    /**
     * Builds the device profile map. Keys and value types must stay in lockstep
     * with `lib/src/device/arch.dart` (`ArchDetector._fromNativeMap`).
     */
    fun profile(context: Context): Map<String, Any> {
        val isWear = context.packageManager
            .hasSystemFeature(PackageManager.FEATURE_WATCH)
        return mapOf(
            // Ordered best-first by the platform (primary ABI at index 0).
            "supportedAbis" to Build.SUPPORTED_ABIS.toList(),
            "isWearOs" to isWear,
            "cores" to Runtime.getRuntime().availableProcessors(),
            "model" to (Build.MODEL ?: "unknown"),
            "androidSdkInt" to Build.VERSION.SDK_INT,
        )
    }
}
