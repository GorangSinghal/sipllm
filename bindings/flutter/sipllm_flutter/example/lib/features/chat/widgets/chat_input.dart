import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/theme/app_theme.dart';
import '../chat_provider.dart';

class ChatInput extends StatefulWidget {
  const ChatInput({super.key});

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  final _controller = TextEditingController();

  void _send() {
    if (_controller.text.isEmpty) return;
    context.read<ChatProvider>().sendMessage(_controller.text);
    _controller.clear();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      color: AppTheme.background,
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                hintText: "Ask AI...",
                filled: true,
              ),
              onSubmitted: (_) => _send(),
            ),
          ),
          const SizedBox(width: 12),
          IconButton(
            onPressed: _send,
            icon: const Icon(Icons.send_rounded),
            color: AppTheme.primary,
            iconSize: 28,
          ),
        ],
      ),
    );
  }
}