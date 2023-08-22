
def max_phone_numbers_finder(phonebook):
    if not phonebook:
        return None

    # Find the person with the maximum number of phone numbers
    max_phone_count = 0
    person_with_max_phone_nums = None
    for person, phones in phonebook.items():
        phone_nums = len(phones)
        if phone_nums > max_phone_count:
            max_phone_count = phone_nums
            person_with_max_phone_nums = person

    return person_with_max_phone_nums

# Example phonebook
phonebook = {
    'Buster Keatons': ['8877452478', '8747896512', '9988992510'],
    'Charlie Chaplin': ['7452043210', '1235478625'],
    'Harold LLyold': ['9965485333', '8787854529', '1212121212', '9888777799'],
    'Marilyn Monroe': ['988777779', '9999888877', '1112225557'],
}

person_name = max_phone_numbers_finder(phonebook)
print('The Person with maximum phone numbers in their phonebook is {}' .format(person_name))
