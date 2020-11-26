from pymongo import MongoClient
import pprint


def access_specific_collection(collection_name):
    client = MongoClient('localhost', 27017)
    db = client.yelp
    business = db.business
    review = db.review
    tip = db.tip
    checkin = db.checkin
    user = db.user
    collections = {'business': business, 'review': review,
                   'tip': tip, 'checkin': checkin, 'user': user}
    return collections[collection_name]


def access_all_collections():
    """
    Retrieves collections from "yelp" database with mongoDB.

    Returns:
        Tuple of Objects: The five collections from the "yelp" database.
    """
    client = MongoClient('localhost', 27017)
    db = client.yelp
    business = db.business
    review = db.review
    tip = db.tip
    checkin = db.checkin
    user = db.user
    return business, review, tip, checkin, user


def record_counts_examples():
    """
    Prints the record counts and example record of each collection
    in order to explore record structure.
    """
    print('\nRecord Counts:')
    for collection in access_all_collections():
        print(f'{collection.name} records: {collection.estimated_document_count()}')

    print('\nExample Records:')
    for collection in access_all_collections():
        print(f'\n{collection.name}\n------------------------------')
        pprint.pprint(collection.find_one())


if __name__ == "__main__":
    """
    Already loaded the original jsons into the yelp database
    within mongoDB via command line.
    """
    record_counts_examples()
