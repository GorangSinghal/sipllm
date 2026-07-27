import 'package:flutter/material.dart';
import '../../core/models/chat_message.dart';
import '../../core/models/llm_provider.dart';
import '../../core/services/api_service.dart';

class ChatProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();

  List<ChatMessage> messages = [];
  bool isLoading = false;

  LlmType? _selectedType;
  String _currentUrl = '';

  void connect(LlmType type, String url) {
    _selectedType = type;
    _currentUrl = url;
    messages.clear();
    notifyListeners();
  }

  Future<void> sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    messages.add(ChatMessage(text: text, isUser: true));
    isLoading = true;
    notifyListeners();

    try {
      final response = await _apiService.sendMessage(
        baseUrl: _currentUrl,
        prompt: text,
      );

      messages.add(ChatMessage(text: response, isUser: false));
    } catch (e) {
      messages.add(ChatMessage(
        text: "Error: $e",
        isUser: false,
      ));
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }
}