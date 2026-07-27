package com.sipllm.flutter

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.ChannelClient
import com.google.android.gms.wearable.Wearable
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.concurrent.Executors

/**
 * Phone <-> Wear OS model transfer over the Wearable Data Layer.
 *
 * ## Why ChannelClient (not RFCOMM / Bluetooth SPP)
 * Wear OS does not expose classic Bluetooth RFCOMM sockets to apps. The
 * sanctioned transport for paired-device I/O is the Data Layer. Of its three
 * clients — DataClient (small synced items), MessageClient (small one-shot
 * payloads) and ChannelClient (a bidirectional byte *stream*) — only
 * ChannelClient suits a hundreds-of-MB GGUF copy. ChannelClient auto-negotiates
 * the physical link: Bluetooth for the control handshake, and the Wi-Fi
 * High-Bandwidth path for the bulk stream when both devices are on the same
 * network. A channel gives us both an [OutputStream] and an [InputStream], which
 * is what makes the resume handshake below possible in a single connection.
 *
 * ## Resumable protocol (channel path `/sipllm/model-transfer`)
 *  1. Sender opens the channel and writes a newline-terminated JSON header:
 *     `{filename, totalSize, sha256, resumeOffset}`.
 *  2. Receiver reads the header, resolves the destination file, and replies with
 *     a newline-terminated JSON ack `{ackOffset}` = how many valid bytes it
 *     already holds (0 when starting fresh or when resume is impossible).
 *  3. Sender seeks the source file to `ackOffset` and streams the remainder.
 *  4. Receiver appends until it has `totalSize` bytes, then verifies `sha256`.
 *
 * [pause] closes the channel while retaining offsets; the receiver keeps its
 * partial file. [resume] reopens the channel and re-runs the ack handshake so
 * the sender seeks to wherever the receiver actually got to. [cancel] tears the
 * channel down; the receiver discards its partial file on an explicit cancel.
 *
 * All blocking Data Layer / file I/O runs on a single-thread [Executors]
 * executor; progress is marshalled back to the main thread for the EventChannel
 * sink. Correctness beyond compilation depends on real paired hardware and is
 * marked `RUNTIME-UNVERIFIED` at each such site.
 */
class WearModelTransfer(private val context: Context) {

    private companion object {
        const val CHANNEL_PATH = "/sipllm/model-transfer"
        const val BUFFER = 32 * 1024
        const val EMIT_INTERVAL_NS = 200_000_000L // 200 ms progress throttle

        // Wire strings — must match Dart TransferState.name / TransferDirection.name.
        const val DIR_SEND = "send"
        const val DIR_RECEIVE = "receive"
        const val ST_CONNECTING = "connecting"
        const val ST_TRANSFERRING = "transferring"
        const val ST_PAUSED = "paused"
        const val ST_COMPLETED = "completed"
        const val ST_FAILED = "failed"
        const val ST_CANCELED = "canceled"
    }

    private val main = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor { r ->
        Thread(r, "sipllm-wear-transfer").apply { isDaemon = true }
    }

    private val channelClient: ChannelClient = Wearable.getChannelClient(context)
    private val nodeClient = Wearable.getNodeClient(context)

    private var sink: EventChannel.EventSink? = null

    // Cross-thread control flags. Set on the platform (main) thread, read on the
    // io thread's streaming loops.
    @Volatile private var paused = false
    @Volatile private var canceled = false

    /** Retained sender state so a [pause] can be [resume]d. */
    private var send: SendState? = null

    /** Active channel (sender or receiver) so control ops can close it. */
    @Volatile private var activeChannel: ChannelClient.Channel? = null

    private var receiverCallback: ChannelClient.ChannelCallback? = null

    private class SendState(
        val file: File,
        val filename: String,
        val total: Long,
        val sha256: String,
        val nodeId: String,
        val resume: Boolean,
        var offset: Long,
    )

    /** EventChannel stream handler; onListen/onCancel run on the main thread. */
    val streamHandler: EventChannel.StreamHandler = object : EventChannel.StreamHandler {
        override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
            sink = events
        }

        override fun onCancel(arguments: Any?) {
            sink = null
        }
    }

    /** MethodChannel dispatch for `sipllm/wear`. Runs on the main thread. */
    fun handle(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "connectedNodes" -> connectedNodes(result)

            "sendModel" -> {
                val path = call.argument<String>("path")
                if (path == null) {
                    result.error("ARG", "path is required", null)
                    return
                }
                val nodeId = call.argument<String>("nodeId")
                val resume = call.argument<Boolean>("resume") ?: true
                paused = false
                canceled = false
                io.execute { startSend(path, nodeId, resume) }
                result.success(null)
            }

            "listenIncoming" -> {
                io.execute { startReceiver() }
                result.success(null)
            }

            "pause" -> {
                paused = true
                result.success(null)
            }

            "resume" -> {
                val st = send
                if (st == null) {
                    result.error("STATE", "no paused transfer to resume", null)
                } else {
                    paused = false
                    canceled = false
                    io.execute { runSend(st) }
                    result.success(null)
                }
            }

            "cancel" -> {
                canceled = true
                closeActiveChannel()
                result.success(null)
            }

            else -> result.notImplemented()
        }
    }

    /** Releases GMS callbacks and the executor on plugin detach. */
    fun dispose() {
        receiverCallback?.let { channelClient.unregisterChannelCallback(it) }
        receiverCallback = null
        closeActiveChannel()
        io.shutdownNow()
    }

    // ---- node discovery -----------------------------------------------------

    private fun connectedNodes(result: MethodChannel.Result) {
        io.execute {
            try {
                // RUNTIME-UNVERIFIED: requires paired POCO X6 Pro + OnePlus Watch 2
                val nodes = Tasks.await(nodeClient.connectedNodes)
                val list = nodes.map { n ->
                    mapOf(
                        "id" to n.id,
                        "displayName" to n.displayName,
                        "nearby" to n.isNearby,
                    )
                }
                main.post { result.success(list) }
            } catch (e: Exception) {
                main.post { result.error("NODES", e.message, null) }
            }
        }
    }

    private fun pickNodeId(): String? =
        try {
            Tasks.await(nodeClient.connectedNodes).firstOrNull { it.isNearby }?.id
                ?: Tasks.await(nodeClient.connectedNodes).firstOrNull()?.id
        } catch (e: Exception) {
            null
        }

    // ---- sender (phone) -----------------------------------------------------

    private fun startSend(path: String, nodeId: String?, resume: Boolean) {
        val file = File(path)
        val filename = file.name
        emit(progress(DIR_SEND, nodeId, filename, 0, null, 0.0, ST_CONNECTING))
        try {
            if (!file.isFile) throw IllegalStateException("model file not found: $path")
            val total = file.length()
            // sha256 is computed once up front so the receiver can verify the
            // whole file at the end (and revalidate after a resume). This is a
            // full read of the source; acceptable for a one-off transfer.
            val sha = sha256(file)
            val target = nodeId ?: pickNodeId()
                ?: throw IllegalStateException("no connected Wear OS node")
            val st = SendState(file, filename, total, sha, target, resume, 0L)
            send = st
            runSend(st)
        } catch (e: Exception) {
            emit(progress(DIR_SEND, nodeId, filename, 0, null, 0.0, ST_FAILED))
        }
    }

    private fun runSend(st: SendState) {
        var channel: ChannelClient.Channel? = null
        try {
            emit(progress(DIR_SEND, st.nodeId, st.filename, st.offset, st.total, 0.0, ST_CONNECTING))
            // RUNTIME-UNVERIFIED: requires paired POCO X6 Pro + OnePlus Watch 2
            channel = Tasks.await(channelClient.openChannel(st.nodeId, CHANNEL_PATH))
            activeChannel = channel
            val out: OutputStream = Tasks.await(channelClient.getOutputStream(channel))
            val inp: InputStream = Tasks.await(channelClient.getInputStream(channel))

            out.use { output ->
                // 1. header
                val header = JSONObject()
                    .put("filename", st.filename)
                    .put("totalSize", st.total)
                    .put("sha256", st.sha256)
                    .put("resume", st.resume)
                    .put("resumeOffset", st.offset)
                writeLine(output, header.toString())
                output.flush()

                // 2. resume handshake: the receiver tells us how many bytes it
                // already holds so we seek forward instead of restarting.
                var offset = st.offset
                val ack = readLine(inp)
                if (ack != null) {
                    offset = JSONObject(ack).optLong("ackOffset", offset)
                }
                offset = offset.coerceIn(0, st.total)
                st.offset = offset

                // 3. stream the remainder
                val finalState = streamOut(st, output, offset)
                output.flush()
                emit(progress(DIR_SEND, st.nodeId, st.filename, st.offset, st.total, 0.0, finalState))
                if (finalState == ST_COMPLETED || finalState == ST_CANCELED) send = null
            }
        } catch (e: Exception) {
            if (canceled) {
                emit(progress(DIR_SEND, st.nodeId, st.filename, st.offset, st.total, 0.0, ST_CANCELED))
                send = null
            } else {
                // A mid-stream failure leaves `send` intact so a later resume can
                // retry from the last acknowledged offset.
                emit(progress(DIR_SEND, st.nodeId, st.filename, st.offset, st.total, 0.0, ST_FAILED))
            }
        } finally {
            channel?.let { closeChannel(it) }
        }
    }

    /** Streams source bytes from [offset]; returns the terminal wire state. */
    private fun streamOut(st: SendState, out: OutputStream, offset: Long): String {
        RandomAccessFile(st.file, "r").use { raf ->
            raf.seek(offset)
            val buf = ByteArray(BUFFER)
            var sent = offset
            var lastTime = System.nanoTime()
            var lastBytes = sent
            emit(progress(DIR_SEND, st.nodeId, st.filename, sent, st.total, 0.0, ST_TRANSFERRING))
            while (sent < st.total) {
                if (canceled) return ST_CANCELED
                if (paused) return ST_PAUSED
                val want = minOf(BUFFER.toLong(), st.total - sent).toInt()
                val n = raf.read(buf, 0, want)
                if (n <= 0) break
                out.write(buf, 0, n)
                sent += n
                st.offset = sent
                val now = System.nanoTime()
                if (now - lastTime >= EMIT_INTERVAL_NS || sent >= st.total) {
                    val bps = (sent - lastBytes) * 1e9 / (now - lastTime).coerceAtLeast(1)
                    emit(progress(DIR_SEND, st.nodeId, st.filename, sent, st.total, bps, ST_TRANSFERRING))
                    lastTime = now
                    lastBytes = sent
                }
            }
            return if (sent >= st.total) ST_COMPLETED else ST_PAUSED
        }
    }

    // ---- receiver (watch) ---------------------------------------------------

    private fun startReceiver() {
        if (receiverCallback != null) return
        val cb = object : ChannelClient.ChannelCallback() {
            override fun onChannelOpened(channel: ChannelClient.Channel) {
                if (channel.path != CHANNEL_PATH) return
                // RUNTIME-UNVERIFIED: requires paired POCO X6 Pro + OnePlus Watch 2
                io.execute { receive(channel) }
            }
        }
        receiverCallback = cb
        try {
            Tasks.await(channelClient.registerChannelCallback(cb))
        } catch (e: Exception) {
            receiverCallback = null
            emit(progress(DIR_RECEIVE, null, "", 0, null, 0.0, ST_FAILED))
        }
    }

    private fun receive(channel: ChannelClient.Channel) {
        activeChannel = channel
        val nodeId = channel.nodeId
        var filename = ""
        try {
            val inp: InputStream = Tasks.await(channelClient.getInputStream(channel))
            val out: OutputStream = Tasks.await(channelClient.getOutputStream(channel))

            // 1. header
            val headerLine = readLine(inp)
                ?: throw IllegalStateException("empty header")
            val header = JSONObject(headerLine)
            filename = header.getString("filename")
            val total = header.getLong("totalSize")
            val sha = header.optString("sha256", "")
            val resume = header.optBoolean("resume", true)
            emit(progress(DIR_RECEIVE, nodeId, filename, 0, total, 0.0, ST_CONNECTING))

            val dest = File(modelsDir(), filename)
            var have = if (dest.isFile) dest.length() else 0L
            // Discard a stale partial when resume is off, or when its length is
            // impossible for this transfer (longer than the declared total).
            if (!resume || have < 0 || have > total) {
                dest.delete()
                have = 0L
            }

            // 2. ack: report how many valid bytes we already hold.
            val ack = JSONObject().put("ackOffset", have)
            writeLine(out, ack.toString())
            out.flush()

            // 3. append the remainder
            val finalState = streamIn(inp, dest, nodeId, filename, have, total, sha)
            emit(progress(DIR_RECEIVE, nodeId, filename, dest.length(), total, 0.0, finalState))
        } catch (e: Exception) {
            emit(progress(DIR_RECEIVE, nodeId, filename, 0, null, 0.0, ST_FAILED))
        } finally {
            closeChannel(channel)
        }
    }

    private fun streamIn(
        inp: InputStream,
        dest: File,
        nodeId: String,
        filename: String,
        startOffset: Long,
        total: Long,
        sha: String,
    ): String {
        // append=true when resuming so previously received bytes are preserved.
        FileOutputStream(dest, startOffset > 0).use { fos ->
            val buf = ByteArray(BUFFER)
            var written = startOffset
            var lastTime = System.nanoTime()
            var lastBytes = written
            emit(progress(DIR_RECEIVE, nodeId, filename, written, total, 0.0, ST_TRANSFERRING))
            while (written < total) {
                if (canceled) {
                    fos.flush()
                    dest.delete() // explicit cancel discards the partial file
                    return ST_CANCELED
                }
                val n = inp.read(buf)
                if (n < 0) {
                    // Channel closed before completion => the sender paused (or
                    // dropped). Keep the partial file so a resume can continue.
                    fos.flush()
                    return ST_PAUSED
                }
                if (n == 0) continue
                fos.write(buf, 0, n)
                written += n
                val now = System.nanoTime()
                if (now - lastTime >= EMIT_INTERVAL_NS || written >= total) {
                    val bps = (written - lastBytes) * 1e9 / (now - lastTime).coerceAtLeast(1)
                    emit(progress(DIR_RECEIVE, nodeId, filename, written, total, bps, ST_TRANSFERRING))
                    lastTime = now
                    lastBytes = written
                }
            }
            fos.flush()
        }
        // 4. verify integrity end-to-end.
        if (sha.isNotEmpty()) {
            val actual = sha256(dest)
            if (!actual.equals(sha, ignoreCase = true)) {
                dest.delete()
                return ST_FAILED
            }
        }
        return ST_COMPLETED
    }

    private fun modelsDir(): File =
        File(context.filesDir, "models").apply { mkdirs() }

    // ---- helpers ------------------------------------------------------------

    private fun closeActiveChannel() {
        activeChannel?.let { closeChannel(it) }
    }

    private fun closeChannel(channel: ChannelClient.Channel) {
        try {
            Tasks.await(channelClient.close(channel))
        } catch (_: Exception) {
            // Already closed / node gone — nothing to do.
        }
        if (activeChannel === channel) activeChannel = null
    }

    private fun emit(event: Map<String, Any?>) {
        main.post { sink?.success(event) }
    }

    private fun progress(
        direction: String,
        nodeId: String?,
        filename: String,
        sent: Long,
        total: Long?,
        bytesPerSecond: Double,
        state: String,
    ): Map<String, Any?> = mapOf(
        "direction" to direction,
        "nodeId" to nodeId,
        "filename" to filename,
        "sent" to sent,
        "total" to total,
        "bytesPerSecond" to bytesPerSecond,
        "state" to state,
    )

    /** Writes a UTF-8 line terminated by '\n' (used for the header/ack). */
    private fun writeLine(out: OutputStream, line: String) {
        out.write(line.toByteArray(Charsets.UTF_8))
        out.write('\n'.code)
    }

    /** Reads a single '\n'-terminated UTF-8 line, or null at EOF. */
    private fun readLine(inp: InputStream): String? {
        val bos = ByteArrayOutputStream()
        while (true) {
            val b = inp.read()
            if (b < 0) return if (bos.size() == 0) null else bos.toString("UTF-8")
            if (b == '\n'.code) break
            bos.write(b)
        }
        return bos.toString("UTF-8")
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { ins ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = ins.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
