from chatAssist import ChatAssistant
from Slice import Slice


def main():
    # API KEY
    key = 'LL-CXhk0XQPqxoMcaTSPtMpPTer4b1cLqCFoRjxbZrxZNYugeWTTPngb6oavKrpFd00'
    # change it to long.txt for testing longer tokens
    doc = Slice('fit.txt')

    print('Slicing the text...\n')

    slices = doc.slice_document(2048)

    print(f'The document is composed of {doc.token_count} tokens.')
    print(f'It exceeds the context window.') if doc.token_count > 2048 else print(f'It fits the context window.')

    assistant = ChatAssistant(api_key=key)

    for index, slice_text in enumerate(slices, start=1):
        print(f'\nGenerating answer for slice {index}...')
        answer = assistant.generate_answer(slice_text)
        print(f'---SLICE{index}---\n {answer}')


if __name__ == "__main__":
    main()

