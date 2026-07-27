import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import 'chat_provider.dart';
import 'widgets/chat_bubble.dart';
import 'widgets/chat_input.dart';

class ChatScreen extends StatelessWidget {
  const ChatScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Session Active"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: Consumer<ChatProvider>(
              builder: (context, provider, _) {
                if (provider.messages.isEmpty && !provider.isLoading) {
                  return const Center(
                    child: Text(
                      "System Ready.\nWaiting for input...",
                      textAlign: TextAlign.center,
                      style: TextStyle(color: AppTheme.textSec),
                    ),
                  );
                }

                return ListView.builder(
                  padding: const EdgeInsets.only(top: 16),
                  itemCount: provider.messages.length + (provider.isLoading ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (index == provider.messages.length) {
                      return const Center(
                        child: Padding(
                          padding: EdgeInsets.all(16.0),
                          child: CircularProgressIndicator(color: AppTheme.primary),
                        ),
                      );
                    }
                    return ChatBubble(message: provider.messages[index]);
                  },
                );
              },
            ),
          ),
          const ChatInput(),
        ],
      ),
    );
  }
}