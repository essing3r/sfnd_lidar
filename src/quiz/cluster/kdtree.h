/* \author Aaron Brown */
// Quiz on implementing kd tree

#include "../../render/render.h"

// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node *left;
	Node *right;

	Node(std::vector<float> arr, int setId)
		: point(arr), id(setId), left(NULL), right(NULL)
	{
	}
};

struct NodeSearchQuery
{
	std::vector<float> point;
	std::vector<std::pair<float, float>> box;
	float radius;

	NodeSearchQuery(const std::vector<float> &point, float radius)
		: point(point), radius(radius)
	{

		box.resize(point.size());
		for (size_t i = 0; i < point.size(); ++i)
		{
			box[i].first = point[i] - radius;
			box[i].second = point[i] + radius;
		}
	}

	inline bool PointInsideRadius(const std::vector<float> &other) const
	{
		float distance_squared = 0.;
		for (size_t i = 0; i < std::min(point.size(), other.size()); ++i)
		{
			distance_squared += (point[i] - other[i]) * (point[i] - other[i]);
		}
		return std::sqrt(distance_squared) <= radius;
	}

	inline bool PointInsideBox(const std::vector<float> &other) const
	{
		bool inside_box = true;
		for (size_t i = 0; i < std::min(point.size(), other.size()); ++i)
		{
			inside_box &= !(other[i] > box[i].second) && !(other[i] < box[i].first);
		}
		return inside_box;
	}
};

struct KdTree
{
	Node *root;

	KdTree()
		: root(NULL)
	{
	}

	void insert(std::vector<float> point, int id)
	{
		Node *new_node = new Node(point, id);
		insertNodeRecursive(new_node, &root, 0);
	}

	void insertNodeRecursive(Node *node, Node **tree, int depth)
	{
		if (*tree == NULL)
		{
			*tree = node;
			return;
		}

		const size_t index = depth % node->point.size();
		const bool node_is_less_than = (node->point[index] < (*tree)->point[index]);
		tree = (node_is_less_than ? &(*tree)->left : &(*tree)->right);
		return insertNodeRecursive(node, tree, depth + 1);
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
		NodeSearchQuery query(target, distanceTol);
		searchNodeRecursive(root, query, 0, &ids);
		return ids;
	}

	void searchNodeRecursive(const Node *node, const NodeSearchQuery &query, size_t depth, std::vector<int> *ids)
	{
		if (node == NULL)
			return;

		if (query.PointInsideBox(node->point))
		{
			if (query.PointInsideRadius(node->point))
			{
				ids->push_back(node->id);
			}
		}

		const size_t index = depth % query.point.size();
		if (query.box[index].first < node->point[index])
			searchNodeRecursive(node->left, query, depth + 1, ids);
		if (query.box[index].second > node->point[index])
			searchNodeRecursive(node->right, query, depth + 1, ids);
	}
};
