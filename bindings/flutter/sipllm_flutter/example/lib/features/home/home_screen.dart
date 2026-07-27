import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/models/llm_provider.dart';
import '../../core/theme/app_theme.dart';
import '../chat/chat_provider.dart';
import '../chat/chat_screen.dart';
import 'widgets/provider_card.dart';
import 'widgets/info_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  static const String _hiddenUrl = "https://bimestrial-nuggety-dominick.ngrok-free.dev";

  void _connect(BuildContext context, LlmType type) {
    context.read<ChatProvider>().connect(type, _hiddenUrl);
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ChatScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned(
            top: -100,
            right: -100,
            child: ImageFiltered(
              imageFilter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
              child: Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: AppTheme.primary.withOpacity(0.15),
                ),
              ),
            ),
          ),

          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 30),

                  const Text(
                    "NEURAL\nNETWORK\nHUB",
                    style: TextStyle(
                      fontSize: 42,
                      fontWeight: FontWeight.w900,
                      color: Colors.white,
                      height: 1.1,
                      letterSpacing: -1,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    "Select an active node to establish uplink.",
                    style: TextStyle(
                      color: AppTheme.textSec.withOpacity(0.8),
                      fontSize: 16,
                    ),
                  ),

                  const SizedBox(height: 40),

                  Expanded(
                    child: ListView.builder(
                      physics: const BouncingScrollPhysics(),
                      itemCount: LlmType.values.length,
                      itemBuilder: (context, index) {
                        return ProviderCard(
                          type: LlmType.values[index],
                          onTap: () => _connect(context, LlmType.values[index]),
                        );
                      },
                    ),
                  ),

                  const Padding(
                    padding: EdgeInsets.only(bottom: 20, top: 10),
                    child: InfoCard(),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}