import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  Future<String> sendMessage({
    required String baseUrl,
    required String prompt,
  }) async {
    String cleanUrl = baseUrl.trim();
    if (cleanUrl.endsWith('/')) cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    if (cleanUrl.endsWith('/v1')) cleanUrl = cleanUrl.substring(0, cleanUrl.length - 3);

    final uri = Uri.parse('$cleanUrl/v1/chat/completions');

    try {
      final response = await http.post(
        uri,
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer no-key",
          "ngrok-skip-browser-warning": "true",
        },
        body: jsonEncode({
          "model": "local-model",
          "messages": [
            {"role": "user", "content": prompt}
          ],
          "temperature": 0.7
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices'][0]['message']['content'];
      } else {
        throw Exception("Error ${response.statusCode}: ${response.body}");
      }
    } catch (e) {
      throw Exception("Connection failed: $e");
    }
  }
}