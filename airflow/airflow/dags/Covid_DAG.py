from airflow.operators.python import PythonOperator
import datetime
import time
from airflow import DAG
from airflow.utils.dates import days_ago


dag = DAG(
    dag_id='Covid_learning',
    description='DAG to execute learning from new data',
    tags=['Covid_project', 'datascientest'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(2),
    }
)


# definition of the function to execute
def load_data_learning():
    print('Load for learning')
    time.sleep(10)

def execute_learning():
    print("Learning ...")
    time.sleep(10)

def load_data_test():
    print("load data for test")
    time.sleep(10)

def execute_test():
    print('execute test for performance mesurement')
    time.sleep(10)

def read_perf():
    print('read perf')
    time.sleep(10)

def valid_new_model():
    print('valid new model')
    time.sleep(10)

learn1 = PythonOperator(
    task_id='load_data_learning',
    doc_md="""
    This task reads the new data and makes it available for learning process
    """,
    python_callable=load_data_learning,
    dag=dag
)

learn2 = PythonOperator(
    task_id='execute_learning',
    doc_md="""
    This task starts the relearning process
    """,
    python_callable=execute_learning,
    dag=dag
)

test1 = PythonOperator(
    task_id='load_data_test',
    doc_md="""
    This task reads the data and makes it available for test process
    """,
    python_callable=load_data_test,
    dag=dag
)

test2 = PythonOperator(
    task_id='execute_test',
    doc_md="""
    This task starts the test process
    """,
    python_callable=execute_test,
    dag=dag
)

test3 = PythonOperator(
    task_id='read_performance',
    doc_md="""
    This task reads the performance
    """,
    python_callable=read_perf,
    dag=dag
)

valid = PythonOperator(
    task_id='validation_performance',
    doc_md="""
    This task valide the new model for the exploitation process
    """,
    python_callable=valid_new_model,
    dag=dag
)

learn1>>[learn2, test1]
[learn2,test1]>>test2>>test3>>valid