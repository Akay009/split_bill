import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
import matplotlib.pyplot as plt
from config import *



# MongoDB setup (replace with your MongoDB URI and database name)
client = MongoClient(mongo_url)
db = client["split_bill_db"]
expenses_collection = db["expenses"]





# Function to add a new expense to MongoDB
def add_expense(user_name, amount, description):
    expense = {
        "user_name": user_name,
        "amount": amount,
        "description": description,
        "date": pd.Timestamp.now()
    }
    expenses_collection.insert_one(expense)

# Function to get all expenses from MongoDB
def get_all_expenses():
    expenses = list(expenses_collection.find())
    # Convert ObjectId to string
    for expense in expenses:
        expense['_id'] = str(expense['_id'])
    return pd.DataFrame(expenses)


def delete_expense(expense_id):
    expenses_collection.delete_one({"_id": ObjectId(expense_id)})

# Function to update an expense
def update_expense(expense_id, user_name, amount, description):
    expenses_collection.update_one(
        {"_id": ObjectId(expense_id)},
        {"$set": {
            "user_name": user_name,
            "amount": amount,
            "description": description
        }}
    )

# Function to delete all expenses
def delete_all_expenses():
    expenses_collection.delete_many({})

# Streamlit App
st.title("Split Bill App")

# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", ["Add Expense", "View Expenses", "Manage Expenses", "Expense Graphs"])

# Page for adding a new expense
if page == "Add Expense":
    st.header("Add a new expense")
    user_name = st.selectbox("Select your name", ["Kunal", "Himanshu", "Aakash"])

    # Input fields for expense details
    amount = st.number_input("Amount (in INR)", min_value=0.0, format="%.2f")
    description = st.text_input("Description")

    # Submit button to add the expense
    if st.button("Add Expense"):
        if user_name and amount > 0 and description:
            add_expense(user_name, amount, description)
            st.success("Expense added successfully!")
        else:
            st.error("Please enter valid details.")

# Page for viewing expenses
if page == "View Expenses":
    st.header("Current Expenses")

    expenses = get_all_expenses()

    if not expenses.empty:
        st.subheader("All Expenses")
        st.dataframe(expenses)

        total_expenses = expenses["amount"].sum()
        num_users = len(expenses["user_name"].unique())
        equal_share = total_expenses / num_users

        users_expenses = expenses.groupby("user_name")["amount"].sum()

        st.subheader(f"Total Bill: ₹{total_expenses:.2f}")
        st.subheader("Each User Should Receive/Pay:")

        # Calculate balances
        balances = {user: total_spent - equal_share for user, total_spent in users_expenses.items()}

        # Track payments
        payments = []

        for user, balance in balances.items():
            if balance > 0:
                st.write(f"{user} should receive: ₹{balance:.2f}")
            elif balance < 0:
                st.write(f"{user} owes: ₹{-balance:.2f}")
            else:
                st.write(f"{user} is settled.")

        # Payments calculation
        total_to_pay = sum(-balance for balance in balances.values() if balance < 0)  # Total amount owed
        total_to_receive = sum(balance for balance in balances.values() if balance > 0)  # Total amount to receive

        # Distribute payments proportionally
        for user, balance in balances.items():
            if balance < 0:  # Users who owe money
                amount_to_pay = -balance  # Amount this user owes
                # Distribute the amount owed to those who should receive
                for receiver, receive_balance in balances.items():
                    if receive_balance > 0:  # Only those who should receive money
                        amount_receiver = (receive_balance / total_to_receive) * amount_to_pay
                        payments.append((user, receiver, amount_receiver))

        # Display payments
        for payer, receiver, amount in payments:
            st.write(f"{payer} will pay ₹{amount:.2f} to {receiver}")
    else:
        st.write("No expenses added yet.")

if page == "Manage Expenses":
    st.header("Manage Expenses")

    # Display all entries
    expenses = get_all_expenses()

    if not expenses.empty:
        st.subheader("All Expenses")
        st.dataframe(expenses)

        # Option to delete an entry
        st.subheader("Delete a Specific Entry")
        expense_to_delete = st.selectbox("Select an expense to delete", expenses['_id'].values)

        if st.button("Delete Selected Expense"):
            delete_expense(expense_to_delete)
            st.success("Expense deleted successfully!")

        # Option to modify an entry
        st.subheader("Modify an Entry")
        expense_to_modify = st.selectbox("Select an expense to modify", expenses['_id'].values)
        selected_expense = expenses[expenses['_id'] == expense_to_modify].iloc[0]

        new_user_name = st.selectbox("Select your name", ["Kunal", "Himanshu", "Aakash"],
                                     index=["Kunal", "Himanshu", "Aakash"].index(selected_expense['user_name']))
        new_amount = st.number_input("New Amount (in INR)", min_value=0.0, value=selected_expense['amount'],
                                     format="%.2f")
        new_description = st.text_input("New Description", value=selected_expense['description'])

        if st.button("Update Expense"):
            update_expense(expense_to_modify, new_user_name, new_amount, new_description)
            st.success("Expense updated successfully!")

        # Option to delete all entries
        if st.button("Delete All Expenses"):
            delete_all_expenses()
            st.success("All expenses deleted successfully!")
    else:
        st.write("No expenses to manage.")



if page == "Expense Graphs":
    st.header("Expense Graphs")

    # Get expenses
    expenses = get_all_expenses()

    if not expenses.empty:
        # Expenses by user
        st.subheader("Total Expenses by User")
        expenses_by_user = expenses.groupby("user_name")["amount"].sum().reset_index()

        # Plotting expenses by user
        plt.figure(figsize=(10, 5))
        plt.bar(expenses_by_user['user_name'], expenses_by_user['amount'], color='skyblue')
        plt.xlabel("User Name")
        plt.ylabel("Total Amount (in INR)")
        plt.title("Total Expenses by User")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Expenses by description
        st.subheader("Total Expenses by Description")
        expenses_by_description = expenses.groupby("description")["amount"].sum().reset_index()

        # Plotting expenses by description
        plt.figure(figsize=(10, 5))
        plt.bar(expenses_by_description['description'], expenses_by_description['amount'], color='lightgreen')
        plt.xlabel("Description")
        plt.ylabel("Total Amount (in INR)")
        plt.title("Total Expenses by Description")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    else:
        st.write("No expenses to plot.")