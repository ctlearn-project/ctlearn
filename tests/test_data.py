import pytest

from ctalearn.data import gen_fn_HDF5, split_indices_lists

"""
@pytest.fixture(scope='session')
def example_HDF5_files(tmpdir_factory):
   

    h5_file_paths = [file1,file2]

    return h5_file_paths

def test_synchronized_open_file(h5_file_path): 

def test_synchronized_close_file(h5_file_path):

"""

def test_gen_fn_HDF5():
    
    file_list = ["test_file1.h5","test_file2.h5"]
    indices_by_file = [[1,2],[3,4]]

    generator = gen_fn_HDF5(file_list,indices_by_file)

    assert next(generator) == (b"test_file1.h5",1)
    assert next(generator) == (b"test_file1.h5",2)
    assert next(generator) == (b"test_file2.h5",3)
    assert next(generator) == (b"test_file2.h5",4)


def test_split_indices_lists():
    
    indices_lists = [[5,9,3,2,1,4,9,2,11,14,15],[1,3,5,7]]
    validation_split = 0.1
    
    training_lists, validation_lists = split_indices_lists(indices_lists,validation_split)

    for i, (train_list, val_list) in enumerate(zip(training_lists, validation_lists)):
        assert len(train_list) + len(val_list) == len(indices_lists[i])
        
        indices_before = set(indices_lists[i])
        indices_after = set(train_list + val_list)

        assert indices_before == indices_after

