
def fix_typos_in_wake_word(sentence, wake_words, wake_word):
    words = sentence.split()  # split the sentence into words
    
    for i in range(len(words)):
        if words[i] in wake_words:
            words[i] = wake_word  # replace the given word with the key word
    formatted_sentence = ' '.join(words)  # join the words back into a sentence

    return formatted_sentence


def is_user_talking_to_me(transcript, wake_words):
    words = transcript.split()  # split the sentence into words
    for word in words:
        if word in wake_words:
            return True
    return False

def update_conversation(current_conversation, role, content):
    message = {"role": role, "content": content}
    current_conversation.append(message)
