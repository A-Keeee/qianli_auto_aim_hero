#pragma once
#include <iostream>
#include <map>
#include <queue>

#include "opencv2/opencv.hpp"

template<class K, class V>
class FixedSizeMapQueue 
{
public:
    FixedSizeMapQueue();
    void set_capacity(std::size_t capacity);
    void insert(const K& key, const V& value);
    void pop();
    V get(const K& key);
    void change(const K& key, const V& value);
    bool contains(const K& key);
    std::size_t get_size();
    std::pair<K, V> get_begin();
    std::pair<K, V> get_end();

    std::map<K, V> map_;
    std::queue<K> order_;
    size_t capacity_;
};

template<class K, class V>
FixedSizeMapQueue<K, V>::FixedSizeMapQueue()
{
    std::cout << "FixedSizeMapQueue constructed" << std::endl;
    capacity_ = 20;
}

template<class K, class V>
void FixedSizeMapQueue<K, V>::set_capacity(std::size_t capacity)
{
    capacity_ = capacity;
}

template<class K, class V>
void FixedSizeMapQueue<K, V>::insert(const K& key, const V& value)
{
    if(map_.find(key) != map_.end())
    {
        map_[key] = value;
        return;
    }
    if(map_.size() >= capacity_)
    {
        K key_to_remove = order_.front();
        order_.pop();
        map_.erase(key_to_remove);
    }
    map_[key] = value;
    order_.push(key);
}

template<class K, class V>
void FixedSizeMapQueue<K, V>::pop()
{
    if(order_.empty())
    {
        return;
    }
    K key_to_remove = order_.front();
    order_.pop();
    map_.erase(key_to_remove);
}

template<class K, class V>
V FixedSizeMapQueue<K, V>::get(const K& key)
{
    auto it = map_.find(key);
    if (it != map_.end()) {
        return it->second;
    } else {
        throw std::out_of_range("Key not found");
    }
}

template<class K, class V>
void FixedSizeMapQueue<K, V>::change(const K& key, const V& value)
{
    if(map_.find(key) != map_.end())
    {
        map_[key] = value;
    }
    else
    {
        std::cout << "The key is not in the map_" << std::endl;
        return;
    }
}

template<class K, class V>
bool FixedSizeMapQueue<K, V>::contains(const K& key)
{
    return map_.find(key) != map_.end();
}

template<class K, class V>
std::size_t FixedSizeMapQueue<K, V>::get_size()
{
    return map_.size();
}

template<class K, class V>
std::pair<K, V> FixedSizeMapQueue<K, V>::get_begin()
{
    return std::make_pair(order_.front(), map_[order_.front()]);
}

template<class K, class V>
std::pair<K, V> FixedSizeMapQueue<K, V>::get_end()
{
    return std::make_pair(order_.back(), map_[order_.back()]);
}